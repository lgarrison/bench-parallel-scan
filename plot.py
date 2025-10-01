import click
import matplotlib.pyplot as plt
from astropy.table import Table, vstack


@click.command()
@click.argument("csv_files", nargs=-1, type=click.Path(exists=True))
@click.option("--rate", "-r", is_flag=True, help="Plot rate instead of time")
def plot_benchmark_results(csv_files, rate=False):
    if not csv_files:
        raise click.UsageError("Provide at least one CSV file.")

    tables = [Table.read(path, format="ascii.ecsv") for path in csv_files]
    combined_table = tables[0] if len(tables) == 1 else vstack(tables)

    fig, ax = plt.subplots()

    methods = sorted(set(combined_table["method"]))

    for method in methods:
        subset = combined_table[combined_table["method"] == method]
        subset.sort("N")
        y_values = subset["N"] / subset["time"] if rate else subset["time"]
        ax.plot(subset["N"], y_values, label=method, marker="o")

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel("Array Size (N)")
    ax.set_ylabel("Rate (elements/second)" if rate else "Time (seconds)")
    ax.set_title("Matmul Scan Benchmark Results" + (" (Rate)" if rate else ""))

    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend(title="Implementation")

    fig.tight_layout()
    output_name = "benchmark_rate.png" if rate else "benchmark_comparison.png"
    fig.savefig(output_name, dpi=144)
    print(f"Plot saved as {output_name}")
    plt.show()


if __name__ == "__main__":
    plot_benchmark_results()
