import math
import plotly.graph_objects as go


def main():
    # Generate data for an exponential curve
    x = [i * 0.1 for i in range(51)]  # 0.0 to 5.0
    y = [math.exp(val) for val in x]

    # Create the plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="e^x"))
    fig.update_layout(
        title="Exponential Function",
        xaxis_title="x",
        yaxis_title="e^x",
    )

    # Save as a PNG image
    fig.write_image("exponential.png")
    print("Plot saved to exponential.png -- your repo setup is working!")


if __name__ == "__main__":
    main()
