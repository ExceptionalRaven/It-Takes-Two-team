import os
import json
import asyncio
import logging
import argparse
import traceback
import subprocess
import sys
from pathlib import Path
from openjiuwen.core.foundation.tool import tool

from openjiuwen.core.foundation.llm import Model
from openjiuwen.core.foundation.llm.schema.config import ModelClientConfig, ModelRequestConfig
from openjiuwen.core.single_agent.schema.agent_card import AgentCard
from openjiuwen.harness import create_deep_agent
from openjiuwen.core.runner import Runner
from openjiuwen.core.session.agent import Session


logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Глобальные лимиты и константы
MAX_SESSIONS = 2
INPUT_TO_OUTPUT_RATIO = 10  # 10 входных = 1 выходной
DEFAULT_BUDGET_OUTPUT_TOKENS = 100_000

def get_budget_limit() -> float:
    """Читает лимит выходных токенов из переменной окружения BUDGET."""
    raw_budget = os.getenv("BUDGET", "").strip()
    if not raw_budget:
        return float(DEFAULT_BUDGET_OUTPUT_TOKENS)
    try:
        value = float(raw_budget)
        return value if value > 0 else float(DEFAULT_BUDGET_OUTPUT_TOKENS)
    except ValueError:
        logging.warning(f"Некорректное значение BUDGET='{raw_budget}', используется стандартный лимит {DEFAULT_BUDGET_OUTPUT_TOKENS}.")
        return float(DEFAULT_BUDGET_OUTPUT_TOKENS)

def build_workspace_tools(task_path: str):
    """Создает изолированный набор инструментов для конкретного задания."""
    base_dir = Path(task_path).resolve()
    workspace_dir = base_dir / "workspace"
    target_dir = workspace_dir if workspace_dir.exists() else base_dir

    @tool
    def list_files() -> str:
        """Возвращает список всех файлов в рабочей директории задачи."""
        files = [str(p.relative_to(target_dir)) for p in target_dir.rglob("*") if p.is_file()]
        return "\n".join(files[:100]) if files else "Директория пуста."

    @tool
    def read_file(filename: str, max_lines: int = 120) -> str:
        """Читает первые max_lines строк файла из рабочей директории."""
        file_path = (target_dir / filename).resolve()
        if not str(file_path).startswith(str(target_dir)) or not file_path.is_file():
            return f"Ошибка: файл {filename} не найден."
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                lines = [next(f) for _ in range(max_lines)]
            return "".join(lines)[:3000]
        except StopIteration:
            return file_path.read_text(encoding="utf-8", errors="replace")[:3000]
        except Exception as e:
            return f"Ошибка чтения: {e}"

    @tool
    def run_python(code: str) -> str:
        """Выполняет Python-код в рабочей директории задачи и возвращает stdout/stderr."""
        try:
            proc = subprocess.run(
                [sys.executable, "-c", code],
                cwd=str(target_dir),
                capture_output=True,
                text=True,
                timeout=30
            )
            output = (proc.stdout + "\n" + proc.stderr).strip()
            return output[:2000] if output else "Код выполнен успешно (пустой вывод)."
        except subprocess.TimeoutExpired:
            return "Ошибка: превышен таймаут выполнения (30 сек)."
        except Exception as e:
            return f"Ошибка запуска: {e}"

    @tool
    def run_tests(test_filename: str = "test_mylib.py") -> str:
        """Запускает тестовый файл Python в рабочей директории и возвращает результат."""
        try:
            proc = subprocess.run(
                [sys.executable, test_filename],
                cwd=str(target_dir),
                capture_output=True,
                text=True,
                timeout=30
            )
            output = (proc.stdout + "\n" + proc.stderr).strip()
            return f"Exit code: {proc.returncode}\n{output[:2500]}"
        except Exception as e:
            return f"Ошибка запуска тестов: {e}"

    @tool
    def write_code_file(filename: str, content: str) -> str:
        """Перезаписывает файл с кодом в рабочей директории (не использовать для тестов)."""
        if filename.startswith("test_"):
            return "Ошибка: изменять тестовые файлы запрещено."
        file_path = (target_dir / filename).resolve()
        if not str(file_path).startswith(str(target_dir)):
            return "Ошибка: выход за пределы рабочей директории."
        file_path.write_text(content, encoding="utf-8")
        return f"Файл {filename} успешно обновлен."

    return [list_files, read_file, run_python, run_tests, write_code_file]

def build_workspace_context(task_path: str, category: str = "") -> str:
    """Локально собирает список файлов и превью только для coding и data."""
    workspace_dir = Path(task_path).resolve() / "workspace"
    if not workspace_dir.exists() or category == "search":
        return ""

    files = [
        p for p in workspace_dir.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.name != "answer.txt"
    ]
    if not files:
        return ""

    lines = ["Файлы в текущей рабочей директории (передавай в инструменты ТОЛЬКО имя файла без пути):"]
    for f in files[:15]:
        rel = str(f.relative_to(workspace_dir))
        lines.append(f"- {rel} ({f.stat().st_size} байт)")
        if f.suffix in {".py", ".txt", ".json"} and f.stat().st_size <= 4096:
            content = f.read_text(encoding="utf-8", errors="replace")
            lines.append(f"--- Содержимое {rel} ---\n{content}\n--- Конец {rel} ---")
        elif f.suffix in {".csv", ".tsv"}:
            with open(f, "r", encoding="utf-8", errors="replace") as cf:
                head = "".join([next(cf) for _ in range(4)])
            lines.append(f"--- Первые строки {rel} ---\n{head.strip()}\n---")
    return "\n".join(lines) + "\n"

AGENT_PROFILES = {
    "math": {
        "model": "qwen3.8-27b",
        "max_iterations": 3,
        "system_prompt": (
            "Ты — математический решатель. Все вычисления выполняй строго через инструмент run_python. "
            "Не считай в уме. После получения результата верни ТОЛЬКО итоговое число или строку без пояснений."
        ),
    },
    "coding": {
        "model": "qwen3.8-27b",
        "max_iterations": 5,
        "system_prompt": (
            "Ты — инженер-программист. Код и тесты уже даны в условии. Сразу запусти run_tests. "
            "Если есть ошибки — исправь основной модуль через write_code_file (тесты не трогай) и перезапусти run_tests до Exit code: 0. "
            "В финальном ответе напиши ТОЛЬКО слово DONE."
        ),
    },
    "data": {
        "model": "qwen3.6-35b-a3b-noreason",
        "max_iterations": 4,
        "system_prompt": (
            "Ты — аналитик данных. Используй list_files и read_file (первые 5 строк), чтобы узнать структуру CSV/JSON. "
            "Напиши скрипт и выполни его через run_python. НИКОГДА не печатай таблицы целиком через print(). "
            "Выводи только финальное значение. В конце верни ТОЛЬКО ответ без кода и разметки."
            "Если имя файла уже указано в условии, НЕ вызывай list_files — сразу читай первые 5 строк через read_file или запускай run_python."
        ),
    },
    "search": {
        "model": "qwen3.6-35b-a3b-noreason",
        "max_iterations": 3,
        "system_prompt": (
            "Ты — агент поиска по локальной базе знаний. Вызови list_files, найди нужные файлы через read_file "
            "или поиск скриптом в run_python. Верни ТОЛЬКО краткий точный ответ на вопрос."
        ),
    },
}

class BudgetTracker:
    """Потокобезопасный (в рамках asyncio) трекер потраченных токенов."""

    def __init__(self, limit: float):
        self.spent = 0.0
        self.limit = limit
        # Динамический буфер: 5% от бюджета, но не меньше 50 и не больше 3000 токенов
        self.buffer = min(3000.0, max(50.0, limit * 0.05))
        self.lock = asyncio.Lock()

    async def add_usage(self, input_tokens: int, output_tokens: int):
        async with self.lock:
            cost = output_tokens + (input_tokens / INPUT_TO_OUTPUT_RATIO)
            self.spent += cost
            logging.info(f"Списано токенов: {cost:.2f}. Всего потрачено: {self.spent:.2f} / {self.limit:.0f}")

    async def get_spent(self):
        async with self.lock:
            return self.spent

    async def has_budget(self) -> bool:
        async with self.lock:
            return self.spent < (self.limit - self.buffer)


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

        solvable.append(task)

    logging.info(f"Задач после фильтрации: {len(solvable)} из {len(tasks)}")
    return solvable


def create_agent_for_task(category: str, base_url: str, api_key: str, task_path: str, task_id: str):
    """Создает специализированного агента под категорию задачи."""
    profile = AGENT_PROFILES.get(category, AGENT_PROFILES["search"])
    tools = build_workspace_tools(task_path)

    model = Model(
        model_client_config=ModelClientConfig(
            client_provider="OpenAI",
            api_key=api_key,
            api_base=base_url,
            timeout=300,
        ),
        model_config=ModelRequestConfig(
            model_name=profile["model"],
            temperature=0.1,
        ),
    )

    return create_deep_agent(
        model=model,
        card=AgentCard(
            id=f"solver-{task_id}",
            name=f"Solver_{task_id}",
            description=f"Специализированный агент ({category})",
        ),
        system_prompt=profile["system_prompt"],
        tools=tools,
        cwd=task_path,
        project_root=task_path,
        restrict_to_work_dir=True,
        auto_create_workspace=False,
        enable_sys_operation=False,
        enable_task_loop=False,
        max_iterations=profile["max_iterations"],
        language="ru",
    )

async def verify_and_clean_answer(raw_answer: str, instruction: str, base_url: str, api_key: str, task_id: str) -> str:
    """Второй агент-критик: извлекает чистый финальный ответ, если первый агент выдал мусор или код."""
    cleaned = raw_answer.strip()
    # Если ответ уже короткий и без markdown-кода — не тратим токены на проверку
    if cleaned and "```" not in cleaned and len(cleaned) < 80 and "python" not in cleaned.lower():
        return cleaned

    from openjiuwen.core.runner import Runner
    from openjiuwen.core.session.agent import Session

    verifier_model = Model(
        model_client_config=ModelClientConfig(
            client_provider="OpenAI",
            api_key=api_key,
            api_base=base_url,
            timeout=60,
        ),
        model_config=ModelRequestConfig(
            model_name="qwen3.6-35b-a3b-noreason",
            temperature=0.0,
        ),
    )

    verifier = create_deep_agent(
        model=verifier_model,
        card=AgentCard(id=f"verifier-{task_id}", name=f"Verifier_{task_id}"),
        system_prompt=(
            "Ты — строгий валидатор ответов. Тебе дано условие задачи и сырой вывод агента-решателя. "
            "Извлеки ТОЛЬКО финальный ответ (число, слово или короткую фразу), который нужно записать в answer.txt. "
            "Никакого кода, кавычек, markdown и пояснений."
        ),
        enable_task_loop=False,
        max_iterations=1,
        language="ru",
    )

    req_id = f"verify_{task_id}"
    session = Session(session_id=req_id, card=verifier.card)
    prompt = f"Условие задачи:\n{instruction}\n\nСырой ответ агента:\n{raw_answer}\n\nВыведи только чистый ответ:"

    try:
        await session.pre_run(inputs={"query": prompt, "conversation_id": req_id})
        res = await asyncio.wait_for(
            Runner.run_agent(verifier, {"query": prompt, "conversation_id": req_id}, session=session),
            timeout=60,
        )
        final_text = res.get("content") or res.get("text") or str(res) if isinstance(res, dict) else str(res)
        return final_text.strip().strip("`").strip()
    finally:
        await session.post_run()

async def staggered_start(coro, delay: float):
    if delay > 0:
        await asyncio.sleep(delay)
    return await coro

async def process_task(task, semaphore, budget_tracker, env_vars):
    """Асинхронный воркер для решения одной задачи."""
    async with semaphore:
        if not await budget_tracker.has_budget():
            logging.info(f"Пропуск {task['id']}: бюджет почти исчерпан.")
            return

        logging.info(f"Старт задачи {task['id']} (Категория: {task['category']}, Reward: {task['reward']})")

        agent = create_agent_for_task(task['category'], env_vars['url'], env_vars['key'], task['path'], task['id'])

        try:
            with open(Path(task['path']) / "task.json", 'r', encoding='utf-8') as f:
                task_data = json.load(f)

            ws_context = build_workspace_context(task['path'], task['category'])
            prompt = (
                f"Категория: {task['category']}. Условие: {task_data.get('instruction', '')}\n"
                f"{ws_context}"
                "Работаешь уже внутри директории workspace. Верни ТОЛЬКО финальный ответ без пояснений."
            )

            request_id = f"task_{task['id']}"
            session = Session(session_id=request_id, card=getattr(agent, "card", None))

            try:
                await session.pre_run(inputs={"query": prompt, "conversation_id": request_id})

                result_obj = await asyncio.wait_for(
                    Runner.run_agent(agent, {"query": prompt, "conversation_id": request_id}, session=session),
                    timeout=300
                )

                if isinstance(result_obj, dict):
                    answer = (
                        result_obj.get("output")
                        or result_obj.get("content")
                        or result_obj.get("text")
                        or str(result_obj)
                    )
                    usage = result_obj.get("usage", {})
                    in_tokens = usage.get("input_tokens") or usage.get("prompt_tokens") or (len(prompt) * 2)
                    out_tokens = usage.get("output_tokens") or usage.get("completion_tokens") or max(250, len(answer) * 2)
                else:
                    answer = str(result_obj)
                    in_tokens = len(prompt) * 2
                    out_tokens = max(250, len(answer) * 2)

                # Если агент сам записал ответ в workspace/answer.txt — берем его как эталон
                ws_answer = Path(task['path']) / "workspace" / "answer.txt"
                if ws_answer.exists() and ws_answer.read_text(encoding="utf-8").strip():
                    answer = ws_answer.read_text(encoding="utf-8").strip()
                else:
                    answer = await verify_and_clean_answer(
                        answer, task_data.get('instruction', ''), env_vars['url'], env_vars['key'], task['id']
                    )

                # Сохраняем ответ в корень задачи и дублируем в workspace
                clean_answer = answer.strip()
                (Path(task['path']) / "answer.txt").write_text(clean_answer, encoding="utf-8")
                if (Path(task['path']) / "workspace").exists():
                    ws_answer.write_text(clean_answer, encoding="utf-8")

                await budget_tracker.add_usage(in_tokens, out_tokens)

            finally:
                await session.post_run()

            logging.info(f"Успех {task['id']}. Ответ: '{clean_answer[:50]}'")

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

    # 3. Инициализация трекера и семафора на потоки
    budget_limit = get_budget_limit()
    logging.info(f"Установлен лимит бюджета: {budget_limit:.0f} выходных токенов.")
    budget_tracker = BudgetTracker(budget_limit)
    semaphore = asyncio.Semaphore(MAX_SESSIONS)

    # 4. Запуск пула задач
    tasks_coroutines = [
        staggered_start(process_task(t, semaphore, budget_tracker, env_vars), idx * 1.5 if idx < MAX_SESSIONS else 0.0)
        for idx, t in enumerate(solvable_tasks)
    ]

    await asyncio.gather(*tasks_coroutines)

    final_spent = await budget_tracker.get_spent()
    logging.info(f"Завершено. Итого потрачено токенов (в эквиваленте выходных): {final_spent:.2f}")


if __name__ == "__main__":
    asyncio.run(main())