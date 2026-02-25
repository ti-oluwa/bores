from bores.correlations.core import compute_gas_compressibility_factor



def main():
    z_factor = compute_gas_compressibility_factor(
        pressure=10000,
        temperature=400,
        gas_gravity=0.65,
    )
    print("Z Factor: ", z_factor)


if __name__ == "__main__":
    main()

