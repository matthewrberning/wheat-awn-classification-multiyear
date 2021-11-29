#matthew berning - GWU, 2021
import os
import sys

def yesno(question):
    """
    Simple Yes/No Function.

    input: question - (string) the question to be responded to with (y/n)

    returns True if y/Y
    returns False if n/N

    (otherwise recurses for valid answer)
    from
    """
    prompt = f'{question} (y/n): '
    ans = input(prompt).strip().lower()
    if ans not in ['y', 'n']:
        print(f'{ans} is invalid, please try again...')
        return yesno(question)
    if ans == 'y':
        return True
    return False