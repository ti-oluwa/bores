import marimo

__generated_with = "0.20.2"
app = marimo.App(width="full")


@app.cell
def setup_grid():
    import bores

    cell_dimension = (7.62, 7.62) * bores.c.METERS_TO_FT # Convert from meters to feet
    grid_shape = (100, 1, 20)

    thickness_grid = bores.uniform_grid(grid_shape=grid_shape, value=0.762 * bores.c.METERS_TO_FT)
    return


if __name__ == "__main__":
    app.run()
