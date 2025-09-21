import numpy as np
import control
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

# --- 1. Inicialização do app com tema Bootstrap ---
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.YETI],
    suppress_callback_exceptions=True,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
)
server = app.server


# --- 2. Funções de Planta ---
def criar_planta(tipo="nível"):
    if tipo == "nível":
        A, k = 1.0, 0.5
        return control.TransferFunction([1], [A, k])
    elif tipo == "temperatura":
        K, tau = 2.0, 10.0
        return control.TransferFunction([K], [tau, 1])


# --- 3. Sidebar ---
sidebar = html.Div(
    [
        html.H4("Simulador PID", className="text-white p-2"),
        html.Hr(style={"borderTop": "1px dotted white"}),
        dbc.Nav(
            [dbc.NavLink("Controle de Nível", href="/", active="exact")],
            vertical=True,
            pills=True,
        ),
        html.P("Data Science Academy", className="fixed-bottom text-white p-2"),
    ],
    className="bg-dark",
    style={
        "position": "fixed",
        "top": 0,
        "left": 0,
        "bottom": 0,
        "width": "14rem",
        "padding": "1rem",
    },
)

# --- 4. Layout principal ---
CONTENT_STYLE = {
    "marginLeft": "15rem",
    "marginRight": "1rem",
    "padding": "1rem 2rem",
    "background-color": "#F0F4F5",
}

h_container = dbc.Container(
    [
        dbc.Row(
            [
                # Sliders PID
                dbc.Col(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Label("Kp"),
                                        dcc.Slider(
                                            id="kp-slider",
                                            min=0,
                                            max=20,
                                            step=0.1,
                                            value=3.0,
                                            tooltip={
                                                "placement": "top",
                                                "always_visible": True,
                                            },
                                            marks=None,
                                        ),
                                    ],
                                    width=9,
                                ),
                                dbc.Col(
                                    [
                                        html.Label(""),  # For alignment
                                        dbc.Input(
                                            id="kp-input",
                                            type="number",
                                            value=3.0,
                                            style={"width": "100%"},
                                        ),
                                    ],
                                    width=3,
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Label("Ki"),
                                        dcc.Slider(
                                            id="ki-slider",
                                            min=0,
                                            max=10,
                                            step=0.1,
                                            value=0.5,
                                            tooltip={
                                                "placement": "top",
                                                "always_visible": True,
                                            },
                                            marks=None,
                                        ),
                                    ],
                                    width=9,
                                ),
                                dbc.Col(
                                    [
                                        html.Label(""),
                                        dbc.Input(
                                            id="ki-input",
                                            type="number",
                                            value=0.5,
                                            style={"width": "100%"},
                                        ),
                                    ],
                                    width=3,
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Label("Kd"),
                                        dcc.Slider(
                                            id="kd-slider",
                                            min=0,
                                            max=5,
                                            step=0.05,
                                            value=0.1,
                                            tooltip={
                                                "placement": "top",
                                                "always_visible": True,
                                            },
                                            marks=None,
                                        ),
                                    ],
                                    width=9,
                                ),
                                dbc.Col(
                                    [
                                        html.Label(""),
                                        dbc.Input(
                                            id="kd-input",
                                            type="number",
                                            value=0.1,
                                            style={"width": "100%"},
                                        ),
                                    ],
                                    width=3,
                                ),
                            ]
                        ),
                    ],
                    width=3,
                ),
                # Sistema e Setpoint
                dbc.Col(
                    [
                        html.Label("Sistema"),
                        dcc.Dropdown(
                            id="tipo",
                            options=[
                                {"label": "Nível", "value": "nível"},
                                {"label": "Temperatura", "value": "temperatura"},
                            ],
                            value="nível",
                        ),
                        html.Label("Setpoint"),
                        dcc.Slider(
                            id="setpoint",
                            min=0,
                            max=100,
                            step=1,
                            value=50,
                            tooltip={"placement": "top", "always_visible": True},
                            marks=None,
                        ),
                        html.Br(),
                        dbc.Button(
                            "Iniciar", id="start", color="success", className="me-2"
                        ),
                        dbc.Button(
                            "Parar", id="stop", color="warning", className="me-2"
                        ),
                        dbc.Button("Resetar", id="reset", color="danger"),
                    ],
                    width=3,
                ),
            ],
            className="g-4",
        ),
        dbc.Row(
            dbc.Col(
                dcc.Graph(
                    id="grafico",
                    style={"width": "90%", "height": "600px"},
                    config={"responsive": True},
                )
            )
        ),
        # Intervalo e armazenamento de estado
        dcc.Interval(id="intervalo", interval=100, n_intervals=0, disabled=True),
        dcc.Store(id="sim_state"),
    ],
    fluid=True,
    style=CONTENT_STYLE,
)

# --- 5. Layout geral ---
app.layout = html.Div([dcc.Location(id="url"), sidebar, h_container])


# --- 6. Callback de simulação ---
@app.callback(
    Output("sim_state", "data"),
    Output("intervalo", "disabled"),
    Input("start", "n_clicks"),
    Input("stop", "n_clicks"),
    Input("reset", "n_clicks"),
    Input("intervalo", "n_intervals"),
    State("sim_state", "data"),
    State("kp-slider", "value"),
    State("ki-slider", "value"),
    State("kd-slider", "value"),
    State("setpoint", "value"),
    State("tipo", "value"),
)
def atualizar_sim(
    n_start, n_stop, n_reset, n_intervals, sim_state, kp, ki, kd, setpoint, tipo
):
    dt = 0.1
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

    if sim_state is None or trigger_id == "reset":
        plant_ss = control.ss(criar_planta(tipo))
        X0_plant = np.zeros(plant_ss.nstates)
        y0 = 50.0 if tipo == "nível" else 0.0

        new_state = {
            "running": False,
            "time": [0],
            "y": [y0],
            "sp": [setpoint],
            "u": [0.0],
            "X0_plant": X0_plant.tolist(),
            "tipo": tipo,
            "last_time": 0,
            "I": 0.0,
            "prev_error": 0.0,
        }
        return new_state, True

    if trigger_id == "start":
        sim_state["running"] = True
        return sim_state, False
    if trigger_id == "stop":
        sim_state["running"] = False
        return sim_state, True

    if sim_state["running"]:
        plant_ss = control.ss(criar_planta(tipo))
        X0_plant = np.array(sim_state["X0_plant"])
        y = sim_state["y"][-1]
        t0 = sim_state["last_time"]
        error = setpoint - y

        # --- PID manual SISO ---
        sim_state["I"] += error * dt
        D = (error - sim_state["prev_error"]) / dt
        u = kp * error + ki * sim_state["I"] + kd * D
        u = np.clip(u, 0, 100)
        sim_state["prev_error"] = error

        # Planta
        t_step = [t0, t0 + dt]
        _, y_array, X_plant_out = control.input_output_response(
            plant_ss, t_step, [u], X0_plant, return_x=True
        )
        y = float(y_array[-1])
        X0_plant = X_plant_out[:, -1]

        sim_state["time"].append(t0 + dt)
        sim_state["y"].append(y)
        sim_state["sp"].append(setpoint)
        sim_state["u"].append(u)
        sim_state["X0_plant"] = X0_plant.tolist()
        sim_state["last_time"] = t0 + dt

    return sim_state, not sim_state["running"]


# --- 7. Callback do gráfico ---
@app.callback(Output("grafico", "figure"), Input("sim_state", "data"))
def atualizar_grafico(sim_state):
    fig = go.Figure()
    if sim_state is None or not sim_state["time"]:
        return fig

    time = np.array(sim_state["time"])
    y = np.array(sim_state["y"])
    sp = np.array(sim_state["sp"])
    u = np.array(sim_state["u"])

    # Moving window of 60 seconds
    window_duration = 60.0
    current_time = time[-1]
    start_time = max(0, current_time - window_duration)

    # Find indices within the window
    indices = np.where(time >= start_time)

    time_window = time[indices]
    y_window = y[indices]
    sp_window = sp[indices]
    u_window = u[indices]

    fig.add_trace(
        go.Scatter(
            x=time_window,
            y=y_window,
            mode="lines",
            name="Saída do Processo",
            line=dict(color="blue"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=time_window,
            y=sp_window,
            mode="lines",
            name="Setpoint",
            line=dict(color="red", dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=time_window,
            y=u_window,
            mode="lines",
            name="Abertura Válvula [%]",
            line=dict(color="green"),
            yaxis="y2",
        )
    )

    if sim_state["tipo"] == "nível":
        fig.update_yaxes(title_text="Nível [%]", range=[0, 105])
    else:
        fig.update_yaxes(title_text="Temperatura [°C]", autorange=True)

    fig.update_layout(
        xaxis=dict(
            title="Tempo [s]",
            range=[start_time, current_time if current_time > 0 else 1],
        ),
        yaxis2=dict(title="Abertura [%]", overlaying="y", side="right", range=[0, 105]),
        template="plotly_white",
        autosize=True,
    )
    return fig


# --- 8. Sincronização dos controles PID ---


# Controle para Kp
@app.callback(
    Output("kp-slider", "value"),
    Output("kp-input", "value"),
    Input("kp-slider", "value"),
    Input("kp-input", "value"),
)
def sync_kp(slider_value, input_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # Decide o valor com base no controle que disparou o callback
    value = slider_value if trigger_id == "kp-slider" else input_value

    # Garante que o valor não é negativo
    if value is None or value < 0:
        value = 0.0

    return value, value


# Controle para Ki
@app.callback(
    Output("ki-slider", "value"),
    Output("ki-input", "value"),
    Input("ki-slider", "value"),
    Input("ki-input", "value"),
)
def sync_ki(slider_value, input_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    value = slider_value if trigger_id == "ki-slider" else input_value

    if value is None or value < 0:
        value = 0.0

    return value, value


# Controle para Kd
@app.callback(
    Output("kd-slider", "value"),
    Output("kd-input", "value"),
    Input("kd-slider", "value"),
    Input("kd-input", "value"),
)
def sync_kd(slider_value, input_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    value = slider_value if trigger_id == "kd-slider" else input_value
    if value is None or value < 0:
        value = 0.0

    return value, value


# --- 8. Rodar app ---
if __name__ == "__main__":
    app.run(debug=True)
