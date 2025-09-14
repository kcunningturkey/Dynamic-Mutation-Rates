from src.models import simulate_exponential_fixed

if __name__ == '__main__':
    result = simulate_exponential_fixed(N=100, generations=1000)
    print('Simulation complete. First 10 results:', result[:10])
