import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.collections import LineCollection
import matplotlib.animation as animation
from typing import List, Tuple, Literal, Any, Dict
from functools import cached_property


TITLE = "Fourier Transform Demo"
S_FILE = "signal.mp4"
W_FILE = "winding.mp4"
F_FILE = "fourier.mp4"

FRAME_DATA = Tuple[List[np.ndarray], List[np.ndarray], List[Tuple[int, int]]]


class FourierTransform:
    def __init__(
        self,
        signal_frequency: List[Tuple[Any]],
        signal_time: float,
        winding_frequency: float,
        reference_points: int,
        frames_per_sec: int,
        video_duration: int,
        fourier_mode: Literal["magnitude", "x", "y"],
        plot_centre_of_gravity: bool,
    ) -> None:
        self.signal_time = signal_time
        self.signal_frequency = self.filter_signal_frequency(signal_frequency)
        self.winding_frequency = winding_frequency
        self.reference_points = reference_points
        self.frames_per_sec = frames_per_sec
        self.video_duration = video_duration
        self.plot_centre_of_gravity = plot_centre_of_gravity
        self.fourier_mode = fourier_mode

    @staticmethod
    def filter_signal_frequency(
        signal_frequency: List[Tuple[Any]],
    ) -> List[Tuple[float]]:
        filtered = []
        for f, p, a in signal_frequency:
            try:
                f = float(f)
            except ValueError:
                continue
            try:
                p = float(p)
            except ValueError:
                gr.Warning(
                    f"Error: phase {p} is not float! Assigning 0.0 instead.",
                )
                p = 0.0
            try:
                a = float(a)
            except ValueError:
                gr.Warning(
                    f"Error: amplitude {a} is not float! Assigning 1.0 instead.",
                )
                a = 1.0
            filtered.append((f, p, a))
        return filtered

    @staticmethod
    def calculate_centre_of_gravity(
        mult_signal: List[complex],
    ) -> Tuple[np.ndarray, np.ndarray]:
        x_centre = np.mean([x.real for x in mult_signal])
        y_centre = np.mean([x.imag for x in mult_signal])
        return x_centre, y_centre

    @staticmethod
    def calculate_sum(
        mult_signal: List[complex],
    ) -> Tuple[np.ndarray, np.ndarray]:
        x_sum = np.sum([x.real for x in mult_signal])
        y_sum = np.sum([x.imag for x in mult_signal])
        return x_sum, y_sum

    @staticmethod
    def create_signal(
        frequency: Tuple[float, float, float],
        time: np.ndarray,
    ) -> np.ndarray:
        if frequency is None:
            return 1
        if isinstance(frequency, (int, float)):
            frequency = [(frequency, 0.0, 1.0)]
        return sum(
            [
                a * np.sin(2 * np.pi * (f * time + max(-1.0, min(p, 1.0))))
                for f, p, a in frequency
            ]
        )

    @staticmethod
    def create_winding_coordinate(
        frequency: float,
        time: np.ndarray,
    ) -> np.ndarray:
        if frequency is None:
            return 1
        angle = -2 * np.pi * frequency * time
        return np.cos(angle) + 1j * np.sin(angle)

    @staticmethod
    def get_magnitude(
        point: Tuple[float, float], mode: Literal["magnitude", "x", "y"]
    ) -> float:
        if mode == "magnitude":
            return np.sqrt(point[0] ** 2 + point[1] ** 2)
        elif mode == "x":
            return point[0]
        elif mode == "y":
            return point[1]
        else:
            raise ValueError(
                f"mode {mode} is not one of 'magnitude', 'x' or 'y'!",
            )

    @cached_property
    def time_x(self) -> np.ndarray:
        return np.linspace(0, self.signal_time, self.reference_points)

    @cached_property
    def interval(self) -> int:
        return int(1000 / self.frames_per_sec)

    @cached_property
    def frames(self) -> int:
        return int(self.video_duration * self.frames_per_sec)

    @cached_property
    def _data(self) -> Tuple[FRAME_DATA, np.ndarray]:
        signal = self.create_signal(self.signal_frequency, self.time_x)
        return self.generate_data(signal), signal

    @cached_property
    def data(self) -> Dict[str, Any]:
        (x, y, c), signal = self._data
        return {
            "X": x,
            "Y": y,
            "C": c,
            "S": signal,
            "MXY": max(
                np.absolute(x[-1]).max(),
                np.absolute(y[-1]).max(),
            ),
            "MC": max(np.absolute([ccc for cc in c for ccc in cc])),
            "MS": max(np.absolute(signal)),
        }

    def create_plot(self) -> Tuple[Figure, plt.Axes, Text]:
        fig, ax = plt.subplots(figsize=(15, 10))
        fig.subplots_adjust(
            left=0,
            bottom=0,
            right=1,
            top=1,
            wspace=None,
            hspace=None,
        )
        ax.grid(True)
        ax.spines["left"].set_position("zero")
        ax.spines["right"].set_color("none")
        ax.spines["bottom"].set_position("zero")
        ax.spines["top"].set_color("none")
        ax.margins(x=0, y=0)
        text = ax.text(
            0.85,
            0.9,
            f"Winding Frequency: {self.winding_frequency}",
            transform=ax.transAxes,
        )

        return fig, ax, text

    def generate_data(self, signal: np.ndarray) -> FRAME_DATA:
        X, Y, c_of_gravity = [], [], []
        for frame in range(1, self.frames + 1):
            # create sinusoid and signal
            wrapping_freq = self.create_winding_coordinate(
                self.winding_frequency * frame, self.time_x
            )
            # multiply pure tone and signal
            mult_signal = wrapping_freq * signal

            X.append(np.array([x.real for x in mult_signal]))
            Y.append(np.array([x.imag for x in mult_signal]))

            # calculate and plot centre of gravity
            c_of_gravity.append(
                self.calculate_centre_of_gravity(
                    mult_signal,
                )
            )
        return X, Y, c_of_gravity

    def plot_signal_graph(self) -> Tuple[Figure, Line2D, LineCollection, Text]:
        if not (self.signal_frequency or self.signal_time or self.time_x):
            return None, None, None, None

        vertical_lines = [
            x / self.winding_frequency
            for x in range(int(self.signal_time * self.winding_frequency))
        ]

        fig, ax, text = self.create_plot()
        ax.set_xlim(
            [
                -self.signal_time / 10,
                self.signal_time + self.signal_time / 10,
            ]
        )
        m = self.data["MS"]
        ax.set_ylim([-m - m / 10, m + m / 10])
        (plotted_graph,) = ax.plot(
            self.time_x,
            self.data["S"],
            color="#2779EA",
            linewidth=4,
        )
        plotted_graph2 = ax.vlines(
            vertical_lines,
            ymin=-m,
            ymax=m,
            color="orange",
            linestyles="dashed",
            linewidth=3,
        )
        return fig, plotted_graph, plotted_graph2, text

    def plot_winding_graph(self) -> Tuple[Figure, Line2D, Line2D, Text]:
        if not (self.winding_frequency or self.signal_time or self.time_x):
            return None, None, None, None
        c_graph = None
        fig, ax, text = self.create_plot()
        (graph,) = ax.plot(
            self.data["X"][0],
            self.data["Y"][0],
            color="orange",
            linewidth=3,
            zorder=10,
        )
        if self.plot_centre_of_gravity:
            (c_graph,) = ax.plot(
                [self.data["C"][0][0]],
                [self.data["C"][0][1]],
                marker="o",
                markersize=10,
                color="red",
                zorder=11,
            )
        m = self.data["MXY"]
        ax.set_xlim([-m - m / 10, m + m / 10])
        ax.set_ylim([-m - m / 10, m + m / 10])
        return fig, graph, c_graph, text

    def plot_fourier_transform(self) -> Tuple[Figure, Line2D, Text]:
        fig, ax, text = self.create_plot()
        (graph,) = ax.plot(
            0,
            self.get_magnitude(self.data["C"][0], mode=self.fourier_mode),
            color="#F50247",
            linewidth=3,
            zorder=10,
        )
        x_max = self.winding_frequency * self.frames
        y_max = self.data["MC"]
        ax.set_xlim([-x_max / 10, x_max + x_max / 10])
        ax.set_ylim([-y_max - y_max / 10, y_max + y_max / 10])
        return fig, graph, text

    def update_signal_graph(self, frame: int) -> LineCollection:
        vertical_lines = [
            np.array(
                [
                    [x / (self.winding_frequency * frame), -self.data["MS"]],
                    [x / (self.winding_frequency * frame), self.data["MS"]],
                ]
            )
            for x in range(
                1,
                int(self.signal_time * self.winding_frequency * frame) + 1,
            )
        ]
        self.signal_graph_plotted2.set_segments(vertical_lines)
        self.signal_graph_text.set_text(
            f"Wraping Frequency: {self.winding_frequency*frame:.2f}"
        )
        return self.signal_graph_plotted2

    def update_winding_graph(self, frame: int) -> List[Any]:
        output = []
        self.winding_graph.set_xdata(self.data["X"][frame])
        self.winding_graph.set_ydata(self.data["Y"][frame])
        output.append(self.winding_graph)
        self.winding_graph_text.set_text(
            f"Wraping Frequency: {self.winding_frequency*frame:.2f}"
        )
        if self.plot_centre_of_gravity:
            self.winding_graph_c.set_xdata([self.data["C"][frame][0]])
            self.winding_graph_c.set_ydata([self.data["C"][frame][1]])
            output.append(self.winding_graph_c)

        return output

    def update_fourier_graph(self, frame: int) -> Line2D:
        x = np.linspace(0, self.winding_frequency * frame, frame)
        y = [
            self.get_magnitude(c, mode=self.fourier_mode)
            for c in self.data["C"][:frame]
        ]
        self.fourier_graph.set_xdata(x)
        self.fourier_graph.set_ydata(y)
        self.fourier_graph_text.set_text(
            f"Wraping Frequency: {self.winding_frequency*frame:.2f}"
        )
        return self.fourier_graph

    def create_fourier_animation(
        self, mode: Literal["all", "signal", "winding", "fourier"]
    ) -> List[str]:
        files = []
        if mode == "all" or mode == "signal":
            (
                self.signal_graph_fig,
                self.signal_graph_plotted,
                self.signal_graph_plotted2,
                self.signal_graph_text,
            ) = self.plot_signal_graph()
            ani = animation.FuncAnimation(
                fig=self.signal_graph_fig,
                func=self.update_signal_graph,
                frames=self.frames,
                interval=self.interval,
                repeat=True,
            )
            # f = NamedTemporaryFile("w", suffix=".mp4", delete=False)
            ani.save(S_FILE)
            files.append(S_FILE)
        if mode == "all" or mode == "winding":
            (
                self.winding_graph_fig,
                self.winding_graph,
                self.winding_graph_c,
                self.winding_graph_text,
            ) = self.plot_winding_graph()
            ani2 = animation.FuncAnimation(
                fig=self.winding_graph_fig,
                func=self.update_winding_graph,
                frames=self.frames,
                interval=self.interval,
                repeat=True,
            )
            # f = NamedTemporaryFile("w", suffix=".mp4", delete=False)
            ani2.save(W_FILE)
            files.append(W_FILE)
        if mode == "all" or mode == "fourier":
            (
                self.fourier_graph_fig,
                self.fourier_graph,
                self.fourier_graph_text,
            ) = self.plot_fourier_transform()
            ani3 = animation.FuncAnimation(
                fig=self.fourier_graph_fig,
                func=self.update_fourier_graph,
                frames=self.frames,
                interval=self.interval,
                repeat=True,
            )
            # f = NamedTemporaryFile("w", suffix=".mp4", delete=False)
            ani3.save(F_FILE)
            files.append(F_FILE)

        return files


def update_signal_graph(
    signal_frequency: List[Tuple[Any]],
    signal_time: float,
    reference_points: int,
) -> str:
    animation_obj = FourierTransform(
        signal_frequency=signal_frequency,
        signal_time=signal_time,
        winding_frequency=0,
        reference_points=reference_points,
        frames_per_sec=1,
        video_duration=1,
        fourier_mode="x",
        plot_centre_of_gravity=False,
    )
    return animation_obj.create_fourier_animation(mode="signal")[0]


def generate_animation(
    signal_frequency: List[Tuple[Any]],
    signal_time: float,
    winding_frequency: float,
    reference_points: int,
    frames_per_sec: int,
    video_duration: int,
    fourier_mode: Literal["magnitude", "x", "y"],
    plot_centre_of_gravity: bool,
) -> List[str]:
    animation_obj = FourierTransform(
        signal_frequency=signal_frequency,
        signal_time=signal_time,
        winding_frequency=winding_frequency,
        reference_points=reference_points,
        frames_per_sec=frames_per_sec,
        video_duration=video_duration,
        fourier_mode=fourier_mode,
        plot_centre_of_gravity=plot_centre_of_gravity,
    )
    return animation_obj.create_fourier_animation(mode="all")


with gr.Blocks(title=TITLE) as app:
    with gr.Row():
        with gr.Column(scale=1, variant="default"):
            pass
        with gr.Column(scale=1, variant="default"):
            gr.Markdown(
                f"<h1 style='text-align: center; margin-bottom: 1rem'>{TITLE}</h1>"
            )
        with gr.Column(scale=1, variant="default"):
            pass
    with gr.Row():
        with gr.Column():
            signal_freq_dataframe = gr.Dataframe(
                headers=[
                    "Signal Frequency",
                    "Signal phase",
                    "Signal Amplitude",
                ],
                value=[[3, 0, 1]],
                interactive=True,
                row_count=(1, "dynamic"),
                col_count=(3, "fixed"),
                datatype=["number", "number"],
                type="array",
                label="Signal data",
            )
            signal_time_slider = gr.Slider(
                minimum=0,
                maximum=10,
                value=2,
                interactive=True,
                label="Signal time",
            )
            winding_freq_slider = gr.Slider(
                minimum=0,
                maximum=10,
                value=0.01,
                interactive=True,
                label="Cycling frequency (initial value)",
            )
            video_duration_slider = gr.Slider(
                minimum=0,
                maximum=120,
                value=10,
                interactive=True,
                label="The duration of video",
            )
            frames_per_sec_slider = gr.Slider(
                minimum=0,
                maximum=240,
                value=60,
                interactive=True,
                label="Number of frames per second",
            )
            no_of_reference_points_slider = gr.Slider(
                minimum=0,
                maximum=1_000_000,
                value=100_000,
                step=1,
                interactive=True,
                label="Number of reference points",
            )
            fourier_mode_radio = gr.Radio(
                choices=["magnitude", "x", "y"],
                value="x",
                label="Fourier Mode",
                interactive=True,
            )
            c_of_gravity_checkbox = gr.Checkbox(
                value=True,
                label="Center of Gravity",
            )
            submit_btn = gr.Button(value="Submit", variant="primary")
        with gr.Column():
            signal_graph_video = gr.Video(
                label="Signal graph",
                autoplay=True,
            )
            winding_graph_video = gr.Video(
                label="Winding graph:",
                autoplay=True,
            )
            fourier_graph_video = gr.Video(
                label="Fourier graph:",
                autoplay=True,
            )
    signal_time_slider.change(
        update_signal_graph,
        [
            signal_freq_dataframe,
            signal_time_slider,
            no_of_reference_points_slider,
        ],
        signal_graph_video,
    )
    signal_freq_dataframe.change(
        update_signal_graph,
        [
            signal_freq_dataframe,
            signal_time_slider,
            no_of_reference_points_slider,
        ],
        signal_graph_video,
    )
    no_of_reference_points_slider.change(
        update_signal_graph,
        [
            signal_freq_dataframe,
            signal_time_slider,
            no_of_reference_points_slider,
        ],
        signal_graph_video,
    )
    submit_btn.click(
        generate_animation,
        [
            signal_freq_dataframe,
            signal_time_slider,
            winding_freq_slider,
            no_of_reference_points_slider,
            frames_per_sec_slider,
            video_duration_slider,
            fourier_mode_radio,
            c_of_gravity_checkbox,
        ],
        [signal_graph_video, winding_graph_video, fourier_graph_video],
    )


if __name__ == "__main__":
    app.queue().launch(show_api=False, show_error=True)
