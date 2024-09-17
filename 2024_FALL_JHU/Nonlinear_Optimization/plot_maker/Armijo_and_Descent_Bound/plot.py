import numpy as np
import holoviews as hv
import panel as pn
from holoviews import opts
from holoviews.streams import Stream
import param  # Import param for parameterized classes

# Initialize HoloViews with Bokeh backend
hv.extension('bokeh')

# Define mathematical functions
def quadratic(x, a, b, c):
    return a * x**2 + b * x + c

def Descent_Lemma_Bound(x, f_xk, L, grad_norm):
    a = 0.5 * L * grad_norm**2
    b = -grad_norm ** 2
    c = f_xk
    return quadratic(x, a, b, c)

def linear(x, m, b):
    return m * x + b

def Armijo_Bound(x, f_xk, eta, grad_norm):
    m = -eta * grad_norm**2
    b = f_xk
    return linear(x, m, b)

# Define the plotting function
def plot(eta, L, grad_norm, f_xk):
    x = np.linspace(-1, 5, 300)
    descent_bound = hv.Curve(
        (x, Descent_Lemma_Bound(x, f_xk, L, grad_norm)),
        label='Descent Lemma Bound'
    )
    armijo_bound = hv.Curve(
        (x, Armijo_Bound(x, f_xk, eta, grad_norm)),
        label='Armijo Bound'
    )
    optimal_alpha = 1 / L
    y = np.linspace(-7, 7, 100)
    vline = hv.Curve((optimal_alpha * np.ones(100), y), label='alpha=1/L')
    overlay = descent_bound * armijo_bound * vline

    # Apply all options as keyword arguments
    overlay = overlay.opts(
        tools=['hover'],
        width=600,
        height=400,
        legend_position='right',
        xlabel='step size alpha',
        ylabel='f(x_(k+1))'  # Use LaTeX formatting for mathematical expressions
    )
    return overlay

# Define a Parameterized class to hold the parameters
class Params(param.Parameterized):
    eta = param.Number(default=0.5, bounds=(0.1, 1), doc="Learning rate eta")
    L = param.Number(default=2.0, bounds=(1, 10), doc="Lipschitz constant L")  # Ensure default is float
    grad_norm = param.Number(default=0.5, bounds=(0.1, 5), doc="Gradient norm")
    f_xk = param.Number(default=1.0, constant=True, doc="Function value at xk")  # Ensure default is float

# Instantiate the Parameters
params = Params()

# Create Panel widgets linked to the Parameters
eta_slider = pn.widgets.FloatSlider.from_param(params.param.eta, name='eta')
L_slider = pn.widgets.FloatSlider.from_param(params.param.L, name='L')
grad_norm_slider = pn.widgets.FloatSlider.from_param(params.param.grad_norm, name='grad_norm')

# Define a stream that listens to the Parameters
# Initialize L as a float (2.0) instead of an integer (2)
stream = hv.streams.Stream.define('ParamsStream', eta=0.5, L=2.0, grad_norm=0.5, f_xk=1.0)()

# Callback to update the stream when parameters change
def update_stream(*events):
    stream.event(
        eta=params.eta,
        L=params.L,
        grad_norm=params.grad_norm,
        f_xk=params.f_xk
    )

# Link the Parameterized class to update the stream on changes
params.param.watch(update_stream, ['eta', 'L', 'grad_norm'])

# Initialize the stream with initial parameter values
stream.event(
    eta=params.eta,
    L=params.L,
    grad_norm=params.grad_norm,
    f_xk=params.f_xk
)

# Create the DynamicMap with the plot function and the stream
dmap = hv.DynamicMap(plot, streams=[stream])

# Arrange the layout with the plot and sliders
layout = pn.Column(
    dmap,
    pn.Row(eta_slider, L_slider, grad_norm_slider),
    sizing_mode='stretch_width'
)

# Make the layout servable
layout.servable()
