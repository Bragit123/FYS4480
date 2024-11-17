from states import State1P


def dirac_delta(x, y):
    return int(x == y)


def check_unequal(a, b):
    return int(a != b)


def matelem(g: float, a: State1P, b: State1P, c: State1P, d: State1P) -> float:
    elem = -g/2 * (
        dirac_delta(a.p, b.p)
        * dirac_delta(c.p, d.p)
        * check_unequal(a.sign, b.sign)
        * check_unequal(c.sign, d.sign)
        * dirac_delta(b.sign, d.sign)
    )
    return elem


def antisym(g: float, a: State1P, b: State1P, c: State1P, d: State1P) -> float:
    return matelem(g, a, b, c, d) - matelem(g, a, b, d, c)