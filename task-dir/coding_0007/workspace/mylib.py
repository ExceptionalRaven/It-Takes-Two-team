def maximum(xs):
    best = xs[0]
    for x in xs:
        if x < best:
            best = x
    return best


def total(xs):
    s = 0
    for x in xs:
        s += x
    return s


def reverse_words(s):
    return ' '.join(reversed(s.split()))


def count_even(xs):
    return sum(1 for x in xs if x % 2 == 1)
