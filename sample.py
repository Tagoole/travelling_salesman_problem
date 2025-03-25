def fib_memo(n, memo={}):
    if n in memo:
        return memo[n]  # Return cached result
    if n <= 1:
        return n
    memo[n] = fib_memo(n - 1, memo) + fib_memo(n - 2, memo)  # Store result
    print(f"memo[{n}] = {memo[n]}")  # Print stored result
    return memo[n]

print(fib_memo(10))
