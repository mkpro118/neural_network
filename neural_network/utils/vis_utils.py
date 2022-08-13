from matplotlib import pyplot as plt
from typing import Iterable, Union
import numpy as np

from neural_network.metrics import confusion_matrix


class ConfusionMatrixDisplay:
    def __init__(self, *,
                 labels: Iterable = None,
                 fig_title: str = 'Confusion Matrix',
                 fig_title_fontsize: Union[int, float, tuple] = 18,
                 fig_size: tuple = (8, 8),
                 cmap: str = 'Blues',
                 alpha: float = 0.6,
                 label_fontsize: Union[int, float, tuple] = 14,
                 text_color: str = 'black',
                 alt_text_color: str = 'white',
                 vertical_align: str = 'center',
                 horizontal_align: str = 'center',
                 cmat_fontsize: Union[int, str] = 'xx-large',
                 plt_style: str = 'dark_background'):
        '''
        Plots a confusion matrix using matplotlib

        Parameters
            labels: Iterable, keyword only, default = None
                The labels on the confusion matrix
                If None, will be assigned integers using numpy.arange
                with the shape inferred from the confusion matrix
            fig_title: str, keyword only, default = 'Confusion Matrix'
                Sets a title to the confusion matrix plot. Passed directly to
                pyplot's `suptitle` method
            fig_title_fontsize: Union[int, float, tuple], keyword only, default = 18
                Sets the title fontsize. Passed directly to the `fontsize`
                parameter in pyplot's `suptitle` function
            fig_size: tuple, keyword only, default = (8, 8)
                Sets the size of the plotted figure, passed directly to the
                `figsize` parameter in pyplot's `subplots` function
            cmap: str, keyword only, default = 'Blues'
                The color map for the confusion matrix plot. Passed directly
                to the `cmap` parameter in pyplot's `matshow` function
            alpha: float, keyword only, default = 0.6
                The alpha value for the confusion matrix plot. Passed directly
                to the `alpha` parameter in pyplot's `matshow` function
            label_fontsize: Union[int, float, tuple], keyword only, default = 14
                The fontsize of the tick labels. If int or float, the same is
                passed to both the X-axis and Y-axis. If tuple, the first value
                is passed to the X-axis, and the second value is passed to the
                Y-axis
            text_color: str, keyword only, default = 'black'
                The text color of the off-diagonal elements of the confusion
                matrix.
            alt_text_color: str, keyword only, default = 'white'
                The text color of the diagonal elements of the confusion matrix
            vertical_align: str, keyword only, default = 'center'
                The vertical alignment of the value of the confusion matrix
                elements. Passed directly to the `va` parameter in pyplot's
                `text` function
            horizontal_align: str, keyword only, default = 'center'
                The horizontal alignment of the value of the confusion matrix
                elements. Passed directly to the `ha` parameter in pyplot's
                `text` function
            cmat_fontsize: Union[int, str], keyword only, default = 'xx-large'
                The font size of the confusion matrix values. Passed directly to
                the `size` parameter in pyplot's `text` function
            plt_style: str, default = 'dark_background'
                Sets the style of the plot. Passed directly to pyplot's
                `plt.style.use` method
        '''
        self.labels = labels
        self.fig_title = fig_title
        self.fig_title_fontsize = fig_title_fontsize
        self.fig_size = fig_size
        self.cmap = cmap
        self.alpha = alpha
        self.label_fontsize = label_fontsize
        self.text_color = text_color
        self.alt_text_color = alt_text_color
        self.vertical_align = vertical_align
        self.horizontal_align = horizontal_align
        self.cmat_fontsize = cmat_fontsize
        self.plt_style = plt_style

        if isinstance(self.label_fontsize, (int, float)):
            self.label_fontsize = (self.label_fontsize,) * 2

    def plot_predictions(self, y_true: Union[np.ndarray, tuple, list],
                         y_pred: Union[np.ndarray, tuple, list], *,
                         use_multiprocessing: bool = False):
        '''
        Plots a confusion matrix given the true labels and the predictions

        Paramaters
            y_true: Union[np.ndarray, tuple, list]
                The true labels
            y_pred: Union[np.ndarray, tuple, list]
                The predicted labels
            use_multiprocessing: bool, default = False
                Whether to use multiprocessing while computing
                the confusion matrix
        '''
        cmat = confusion_matrix(y_true, y_pred,
                                use_multiprocessing=use_multiprocessing)
        self.plot_cmat(cmat)

    def plot_cmat(self, cmat: Union[np.ndarray, tuple, list]):
        '''
        Plots a given confusion matrix. Assumes the input is 2 dimensional and
        a square matrix

        Parameters
            cmat: Union[np.ndarray, tuple, list]
                The confusion matrix to plot
        '''

        if self.labels is None or not isinstance(self.labels, Iterable):
            self.labels = np.arange(len(cmat))

        plt.style.use(self.plt_style)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=self.fig_size)

        ax.matshow(cmat, cmap='Blues', alpha=0.6)

        ax.set_xticks(np.arange(len(self.labels)))
        ax.set_xticklabels(map(str, self.labels),
                           fontsize=self.label_fontsize[0])

        ax.set_yticks(np.arange(len(self.labels)))
        ax.set_yticklabels(map(str, self.labels),
                           fontsize=self.label_fontsize[1])

        for i in range(cmat.shape[0]):
            for j in range(cmat.shape[1]):
                ax.text(
                    x=j, y=i, s=cmat[i, j], va=self.vertical_align,
                    ha=self.horizontal_align, size=self.cmat_fontsize,
                    color=self.text_color if i != j else self.alt_text_color
                )

        fig.suptitle(self.fig_title, fontsize=self.fig_title_fontsize)
        plt.show()
