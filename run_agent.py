import os
import json
import asyncio
import logging
import argparse
from pathlib import Path
import traceback

from openjiuwen.core.foundation.llm import Model
from openjiuwen.core.foundation.llm.schema.config import ModelClientConfig, ModelRequestConfig
from openjiuwen.core.single_agent.schema.agent_card import AgentCard
from openjiuwen.harness import create_deep_agent
from openjiuwen.core.runner import Runner
from openjiuwen.core.session.agent import Session



# Попытка импорта openjiuwen (может отличаться в зависимости от документации библиотеки)
try:
    from openjiuwen import Agent, LLMConfig, Tool
except ImportError:
    logging.warning("openjiuwen не установлен. Запуск возможен только в среде с библиотекой.")

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Глобальные лимиты и константы
MAX_SESSIONS = 5
MAX_BUDGET_OUTPUT_TOKENS = 100_000
INPUT_TO_OUTPUT_RATIO = 10  # 10 входных = 1 выходной
SAFE_BUDGET_BUFFER = 3000  # Оставляем резерв, чтобы не превысить лимит


class BudgetTracker:
    """Потокобезопасный (в рамках asyncio) трекер потраченных токенов."""

    def __init__(self, limit):
        self.spent = 0.0
        self.limit = limit
        self.lock = asyncio.Lock()

    async def add_usage(self, input_tokens: int, output_tokens: int):
        async with self.lock:
            cost = output_tokens + (input_tokens / INPUT_TO_OUTPUT_RATIO)
            self.spent += cost
            logging.info(f"Списано токенов: {cost:.2f}. Всего потрачено: {self.spent:.2f} / {self.limit}")

    async def get_spent(self):
        async with self.lock:
            return self.spent

    async def has_budget(self) -> bool:
        async with self.lock:
            return self.spent < (self.limit - SAFE_BUDGET_BUFFER)


def get_sorted_tasks(tasks_dir: str):
    """Парсит все task.json и сортирует задачи по убыванию reward."""
    tasks = []
    base_path = Path(tasks_dir)

    if not base_path.exists():
        logging.error(f"Директория {tasks_dir} не найдена!")
        return tasks

    for entry in os.scandir(base_path):
        if entry.is_dir() and not entry.name.startswith(('.', '__')):
            task_json_path = Path(entry.path) / "task.json"
            if task_json_path.exists():
                try:
                    with open(task_json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        reward = data.get('reward', 0)
                        category = entry.name.split('_')[0]
                        tasks.append({
                            "path": entry.path,
                            "reward": reward,
                            "category": category,
                            "id": entry.name
                        })
                except Exception as e:
                    logging.warning(f"Ошибка чтения {task_json_path}: {e}")

    # Жадная сортировка: сначала самые дорогие
    sorted_tasks = sorted(tasks, key=lambda x: x['reward'], reverse=True)
    logging.info(
        f"Найдено задач: {len(sorted_tasks)}. Максимальный reward: {sorted_tasks[0]['reward'] if sorted_tasks else 0}")
    return sorted_tasks


def filter_solvable_tasks(tasks):
    """
    Классификатор невыполнимых задач.
    В идеале здесь нужно загрузить веса `joblib.load('classifier.pkl')`.
    Пока используем базовую эвристику: отсеиваем пустые или битые директории.
    """
    solvable = []
    for task in tasks:
        workspace_path = Path(task['path']) / "workspace"

        # Эвристика 1: Если workspace нет или он пуст, а задача требует файлов - пропускаем
        if not workspace_path.exists() and task['category'] in ['coding', 'data_analysis', 'search']:
            logging.info(f"Задача {task['id']} отсеяна: отсутствует workspace.")
            continue

        # Здесь можно встроить предсказание ML:
        # features = extract_features(task)
        # if not clf.predict([features])[0]: continue

        solvable.append(task)

    logging.info(f"Задач после фильтрации: {len(solvable)} из {len(tasks)}")
    return solvable


def create_agent_for_task(category: str, base_url: str, api_key: str, task_path: str):
    """Динамический роутинг моделей на основе категории задачи."""
    if category in ["math", "coding"]:
        model_name = "qwen3.8-27b"
    else:
        model_name = "qwen3.6-35b-a3b-noreason"

    model = Model(
        model_client_config=ModelClientConfig(
            client_provider="OpenAI",
            api_key=api_key,
            api_base=base_url,
            timeout=300
        ),
        model_config=ModelRequestConfig(
            model_name=model_name,
            temperature=0.1
        )
    )

    # 2. Создаем агента с изоляцией в директории задачи
    return create_deep_agent(
        model=model,
        card=AgentCard(
            id=f"solver-{category}",
            name=f"Solver_{category}",
            description="Агент для решения задач соревнования"
        ),
        cwd=task_path,                  # Изолируем рабочую директорию
        project_root=task_path,         # Корнем проекта считаем папку задачи
        restrict_to_work_dir=True,      # Запрещаем выход за пределы папки
        enable_task_loop=False,
        max_iterations=5,
        language="ru"
    )
async def process_task(task, semaphore, budget_tracker, env_vars):
    """Асинхронный воркер для решения одной задачи."""
    async with semaphore:
        # Проверяем бюджет перед началом
        if not await budget_tracker.has_budget():
            logging.info(f"Пропуск {task['id']}: бюджет почти исчерпан.")
            return

        logging.info(f"Старт задачи {task['id']} (Категория: {task['category']}, Reward: {task['reward']})")

        agent = create_agent_for_task(task['category'], env_vars['url'], env_vars['key'], task['path'])

        try:
            # Читаем условие из task.json
            with open(Path(task['path']) / "task.json", 'r', encoding='utf-8') as f:
                task_data = json.load(f)

            prompt = f"Реши задачу. Категория: {task['category']}. Условие: {task_data.get('instruction', 'Не указано')}\n"
            prompt += f"Файлы задачи лежат в директории: {Path(task['path']) / 'workspace'}\n"
            prompt += "Верни только финальный ответ, без лишних рассуждений."

            request_id = f"task_{task['id']}"
            session = Session(session_id=request_id, card=getattr(agent, "card", None))

            try:
                await session.pre_run(inputs={"query": prompt, "conversation_id": request_id})

                # Запуск агента через Runner с таймаутом
                result_obj = await asyncio.wait_for(
                    Runner.run_agent(agent, {"query": prompt, "conversation_id": request_id}, session=session),
                    timeout=300
                )

                # В новой архитектуре ответ обычно возвращается словарем или строкой
                if isinstance(result_obj, dict):
                    answer = result_obj.get("content") or result_obj.get("text") or str(result_obj)
                    # Если API не возвращает токены явно, используем эвристику (длина текста / 4)
                    in_tokens = result_obj.get("usage", {}).get("input_tokens", len(prompt) // 4)
                    out_tokens = result_obj.get("usage", {}).get("output_tokens", len(answer) // 4)
                else:
                    answer = str(result_obj)
                    in_tokens = len(prompt) // 4
                    out_tokens = len(answer) // 4

                # Списываем бюджет
                await budget_tracker.add_usage(in_tokens, out_tokens)

            finally:
                # Обязательная очистка сессии агента
                await session.post_run()

            # АТОМАРНОЕ СОХРАНЕНИЕ: Пишем answer.txt
            answer_file = Path(task['path']) / "answer.txt"
            with open(answer_file, "w", encoding="utf-8") as af:
                af.write(answer.strip())

            logging.info(f"Успех {task['id']}. Файл сохранен.")

        except asyncio.TimeoutError:
            logging.error(f"Задача {task['id']} упала по таймауту.")
        except Exception as e:
            logging.error(f"Ошибка в задаче {task['id']}: {e}")
            logging.debug(traceback.format_exc())


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-dir", type=str, required=True, help="Путь к директории с задачами")
    args = parser.parse_args()

    env_vars = {
        'url': os.getenv('LLM_BASE_URL', 'https://api.neuraldeep.ru/v1'),
        'key': os.getenv('LLM_API_KEY', '')
    }

    if not env_vars['key']:
        logging.error("Не задан LLM_API_KEY!")
        return

    # 1. Жадная сортировка
    all_tasks = get_sorted_tasks(args.tasks_dir)

    # 2. Локальный отсев невыполнимых (без API)
    solvable_tasks = filter_solvable_tasks(all_tasks)

    # 3. Инициализация трекера и семафора на 5 потоков
    budget_tracker = BudgetTracker(MAX_BUDGET_OUTPUT_TOKENS)
    semaphore = asyncio.Semaphore(MAX_SESSIONS)

    # 4. Запуск пула задач
    tasks_coroutines = [
        process_task(t, semaphore, budget_tracker, env_vars)
        for t in solvable_tasks
    ]

    await asyncio.gather(*tasks_coroutines)

    final_spent = await budget_tracker.get_spent()
    logging.info(f"Завершено. Итого потрачено токенов (в эквиваленте выходных): {final_spent:.2f}")


if __name__ == "__main__":
    asyncio.run(main())