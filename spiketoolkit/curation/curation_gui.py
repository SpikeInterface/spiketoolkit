import ipywidgets as widgets
from ipywidgets import interact

class CurationGUI(widgets.DOMWidget):
    def __init__(self, sorting):
        '''

        Parameters
        ----------
        sorting
        '''
        widgets.Widget.__init__(self)
        # To prevent automatic figure display when execution of the cell ends
        InlineBackend.close_figures = False

        import matplotlib.pyplot as plt
        import numpy as np

        from IPython.html import widgets
        from IPython.display import display, clear_output

        plt.ioff()
        ax = plt.gca()

        out = widgets.Output()
        button = widgets.Button(description='Next')
        vbox = widgets.VBox(children=(out, button))
        display(vbox)

        def click(b):
            ax.clear()
            ax.plot(np.random.randn(100), np.random.randn(100), '+')
            with out:
                clear_output(wait=True)
                display(ax.figure)

        button.on_click(click)
        click(None)
        pass