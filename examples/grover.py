from qsa import GroverSearch


with GroverSearch(20, [731]) as search:
    print("Logical states:", search.space_size)
    print("Optimal iterations:", search.optimal_iterations)
    search.run_optimal()
    print("Success probability:", search.success_probability)
    print("Sampled basis state:", search.sample())
    print("Engine bytes:", search.estimated_bytes)
