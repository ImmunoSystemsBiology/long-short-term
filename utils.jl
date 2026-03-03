function adjusted_rand_index(cont_table::Matrix{Int})
    # Row and column sums
    a = sum(cont_table, dims=2)  # sum over columns → row sums
    b = sum(cont_table, dims=1)  # sum over rows → column sums
    N = sum(cont_table)           # total number of points

    # Helper: combinations of 2
    comb2(x) = x < 2 ? 0 : x*(x-1)//2  # using rational division to avoid float issues

    # Index: sum of combinations within each cell
    index = sum(comb2.(cont_table))

    # Sums of row and column combinations
    sum_rows = sum(comb2.(a))
    sum_cols = sum(comb2.(b))

    expected_index = sum_rows * sum_cols // comb2(N)
    max_index = (sum_rows + sum_cols) // 2

    # Adjusted Rand Index
    ari = float(index - expected_index) / float(max_index - expected_index)
    return ari
end

# Example usage:
cont_table = [
    94 3 3;
    0 0 100; 
    5 95 0
]

ari = adjusted_rand_index(cont_table)
println("ARI = ", ari)
