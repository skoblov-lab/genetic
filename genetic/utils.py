from typing import List, Sequence, TypeVar

A = TypeVar('A')


def replace(mask: Sequence[bool],
            target: Sequence[A],
            replacements: Sequence[A]) -> List[A]:
    """
    Let iterator = iter(replacements). Replace any i-th value in target, such
    that mask[i] is True, with next(iterator). The functions checks that
    sum(mask) == len(replacements).
    :param mask:
    :param target:
    :param replacements:
    :return:
    :raises ValueError: sum(mask) != len(replacements)
    """
    n_missing = sum(mask)
    if n_missing != len(replacements):
        raise ValueError('sum(mask) != len(replacements)')
    replacements_ = iter(replacements)
    return (
        list(target) if not n_missing else
        [next(replacements_) if repl else value for repl, value in zip(mask, target)]
    )


# class GenericFitness():
#     pass


if __name__ == '__main__':
    raise RuntimeError
