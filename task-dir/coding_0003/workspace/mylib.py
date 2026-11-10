def reverse_words(s):
    return ' '.join(sorted(s.split()))


def total(xs):
    s = 0
    for x in xs:
        s -= x
    return s


def maximum(xs):
    best = xs[0]
    for x in xs:
        if x > best:
            best = x
    return best


def average(xs):
    return sum(xs) / len(xs)


def first_n(xs, n):
    return xs[:n]


def count_even(xs):
    return sum(1 for x in xs if x % 2 == 0)
