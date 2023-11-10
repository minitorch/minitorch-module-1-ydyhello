"""
Collection of the core mathematical operators used throughout the code base.
"""

import math
from typing import Callable, Iterable

# ## Task 0.1
#
# Implementation of a prelude of elementary functions.


def mul(x: float, y: float) -> float:
    "$f(x, y) = x * y$"
    # TODO: Implement for Task 0.1.
    return x * y

    raise NotImplementedError("Need to implement for Task 0.1")


def id(x: float) -> float:
    "$f(x) = x$"
    # TODO: Implement for Task 0.1.
    return x
    raise NotImplementedError("Need to implement for Task 0.1")


def add(x: float, y: float) -> float:
    "$f(x, y) = x + y$"
    # TODO: Implement for Task 0.1.
    return x + y
    raise NotImplementedError("Need to implement for Task 0.1")


def neg(x: float) -> float:
    "$f(x) = -x$"
    # TODO: Implement for Task 0.1.
    return -x
    raise NotImplementedError("Need to implement for Task 0.1")


def lt(x: float, y: float) -> float:
    "$f(x) =$ 1.0 if x is less than y else 0.0"
    # TODO: Implement for Task 0.1.
    return 1.0 if x < y else 0.0
    raise NotImplementedError("Need to implement for Task 0.1")


def eq(x: float, y: float) -> float:
    "$f(x) =$ 1.0 if x is equal to y else 0.0"
    # TODO: Implement for Task 0.1.
    return 1.0 if x == y else 0.0
    raise NotImplementedError("Need to implement for Task 0.1")


def max(x: float, y: float) -> float:
    "$f(x) =$ x if x is greater than y else y"
    # TODO: Implement for Task 0.1.
    return x if x > y else y

    raise NotImplementedError("Need to implement for Task 0.1")


def is_close(x: float, y: float) -> float:
    "$f(x) = |x - y| < 1e-2$"
    # TODO: Implement for Task 0.1.
    return abs(x - y) < 1e-2
    raise NotImplementedError("Need to implement for Task 0.1")


def sigmoid(x: float) -> float:
    r"""
    $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$

    (See https://en.wikipedia.org/wiki/Sigmoid_function )

    Calculate as

    $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$

    for stability.
    """
    # TODO: Implement for Task 0.1.
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))
    raise NotImplementedError("Need to implement for Task 0.1")


def relu(x: float) -> float:
    """
    $f(x) =$ x if x is greater than 0, else 0

    (See https://en.wikipedia.org/wiki/Rectifier_(neural_networks) .)
    """
    # TODO: Implement for Task 0.1.
    return x if x > 0.0 else 0.0
    raise NotImplementedError("Need to implement for Task 0.1")


EPS = 1e-6


def log(x: float) -> float:
    "$f(x) = log(x)$"
    return math.log(x + EPS)


def exp(x: float) -> float:
    "$f(x) = e^{x}$"
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    r"If $f = log$ as above, compute $d \times f'(x)$"
    # TODO: Implement for Task 0.1.
    return d * 1.0 / x
    # raise NotImplementedError("Need to implement for Task 0.1")


def inv(x: float) -> float:
    "$f(x) = 1/x$"
    # TODO: Implement for Task 0.1.
    return 1.0 / x
    # raise NotImplementedError("Need to implement for Task 0.1")


def inv_back(x: float, d: float) -> float:
    r"If $f(x) = 1/x$ compute $d \times f'(x)$"
    # TODO: Implement for Task 0.1.
    if x==0:
        pass
    return -d / x ** 2
    # raise NotImplementedError("Need to implement for Task 0.1")


def relu_back(x: float, d: float) -> float:
    r"If $f = relu$ compute $d \times f'(x)$"
    # TODO: Implement for Task 0.1.
    return d if x > 0 else 0.0
    # raise NotImplementedError("Need to implement for Task 0.1")


# ## Task 0.3

# Small practice library of elementary higher-order functions.


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """
    Higher-order map.

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: Function from one value to one value.

    Returns:
        A function that takes a list, applies `fn` to each element, and returns a
         new list
    """
    # 闭包
    # 捕获了传入的 fn 参数，并可以在以后的调用中使用它
    # TODO: Implement for Task 0.3.
    def myFn(ls: Iterable[float]) -> Iterable[float]:
        return [fn(e) for e in ls]
    return myFn
    # raise NotImplementedError("Need to implement for Task 0.3")


def negList(ls: Iterable[float]) -> Iterable[float]:
    "Use `map` and `neg` to negate each element in `ls`"
    # TODO: Implement for Task 0.3.
    # 将 neg 函数应用到 ls 中的每个元素，以生成一个新的可迭代的序列，其中每个元素都是 ls 中对应元素的相反数
    return map(neg)(ls)
    # raise NotImplementedError("Need to implement for Task 0.3")


def zipWith(
    fn: Callable[[float, float], float], ls1: Iterable[float], ls2: Iterable[float]
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """
    Higher-order zipwith (or map2).

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: combine two values

    Returns:
        Function that takes two equally sized lists `ls1` and `ls2`, produce a new list by
         applying fn(x, y) on each pair of elements.

    """
    # TODO: Implement for Task 0.3.
    ls = [] # 用于存储 fn 函数应用后的结果
    # 使用 zip 函数将 ls1 和 ls2 中对应位置的元素进行配对，然后使用 for 循环遍历这些配对。在循环中，对每一对元素 (x, y) 调用 fn(x, y)，将结果追加到 ls 列表中
    for x, y in zip(ls1, ls2):
        ls.append(fn(x, y))
    return ls
    # raise NotImplementedError("Need to implement for Task 0.3")


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    "Add the elements of `ls1` and `ls2` using `zipWith` and `add`"
    # TODO: Implement for Task 0.3.
    return zipWith(add, ls1, ls2) # 相加
    # raise NotImplementedError("Need to implement for Task 0.3")


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    r"""
    Higher-order reduce.

    Args:
        fn: combine two values
        start: start value $x_0$

    Returns:
        Function that takes a list `ls` of elements
         $x_1 \ldots x_n$ and computes the reduction :math:`fn(x_3, fn(x_2,
         fn(x_1, x_0)))`
    """
    # TODO: Implement for Task 0.3.
    # start是起始值
    # 归约
    # 依次应用 fn 函数将序列中的元素合并在一起
    def myFn(ls: Iterable[float]) -> float:
        # 闭包中并不会修改外部start的值
        t = start
        for e in ls:
            t = fn(e, t) # 使用 fn(e, t) 更新变量 t
        return t
    return myFn
    # raise NotImplementedError("Need to implement for Task 0.3")


def sum(ls: Iterable[float]) -> float:
    "Sum up a list using `reduce` and `add`."
    # TODO: Implement for Task 0.3.
    # 0 是起始值
    return reduce(add, 0)(ls) # 从0开始，依次对元素求和
    # raise NotImplementedError("Need to implement for Task 0.3")


def prod(ls: Iterable[float]) -> float:
    "Product of a list using `reduce` and `mul`."
    # TODO: Implement for Task 0.3.
    return reduce(mul, 1)(ls) # 累积
    # raise NotImplementedError("Need to implement for Task 0.3")