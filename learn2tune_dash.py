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
            [
                dbc.NavLink("Controle de Nível", href="/", active="exact"),
                dbc.NavLink("Controle de Temperatura", href="/temperatura", active="exact"),
            ],
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

# --- 4. Layouts principais para cada página ---
CONTENT_STYLE = {
    "marginLeft": "15rem",
    "marginRight": "1rem",
    "padding": "1rem 2rem",
    "background-color": "#F0F4F5",
}

def create_main_container(tipo="nível"):
    return dbc.Container(
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
                                            html.Label(""),
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
                    # Setpoint e Botões
                    dbc.Col(
                        [
                            html.Label("Setpoint"),
                            dcc.Slider(
                                id="setpoint",
                                min=0,
                                max=100,
                                step=1,
                                value=50 if tipo == "nível" else 70,  # Valor inicial diferente por página
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
app.layout = html.Div(
    [dcc.Location(id="url", refresh=False), sidebar, html.Div(id="page-content")]
)


# --- 6. Callback para renderizar o conteúdo da página com base na URL ---
@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname")
)
def render_page_content(pathname):
    if pathname == "/temperatura":
        return create_main_container("temperatura")
    # Se a URL não for /temperatura, renderiza a página de nível
    return create_main_container("nível")


# --- 7. Callback de simulação ---
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
    State("url", "pathname"),  # Lê o pathname da URL
)
def atualizar_sim(
    n_start, n_stop, n_reset, n_intervals, sim_state, kp, ki, kd, setpoint, pathname
):
    dt = 0.1
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

    # Determina o tipo de sistema com base no pathname da URL
    tipo = "temperatura" if pathname == "/temperatura" else "nível"

    # Previne o erro "TypeError: unsupported operand type(s) for *: 'NoneType' and 'float'"
    kp_val = kp if kp is not None else 0.0
    ki_val = ki if ki is not None else 0.0
    kd_val = kd if kd is not None else 0.0

    if sim_state is None or trigger_id == "reset" or tipo != sim_state.get("tipo"):
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
        u = kp_val * error + ki_val * sim_state["I"] + kd_val * D
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


# --- 8. Callback do gráfico ---
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

    if sim_state.get("tipo") == "nível":
        fig.update_yaxes(title_text="Nível [%]", range=[0, 105])
    else:
        fig.update_yaxes(title_text="Temperatura [°C]", autorange=True)

    fig.update_layout(
        xaxis=dict(title_text="Tempo [s]", range=[start_time, current_time if current_time > 0 else 1]),
        yaxis2=dict(title="Abertura [%]", overlaying="y", side="right", range=[0, 105]),
        template="plotly_white",
        autosize=True,
    )
    return fig


# --- 9. Sincronização dos controles PID e restrição de valor não negativo ---
@app.callback(
    Output("kp-slider", "value"),
    Output("kp-input", "value"),
    Input("kp-slider", "value"),
    Input("kp-input", "value"),
)
def sync_kp(slider_value, input_value):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    value = slider_value if trigger_id == "kp-slider" else input_value
    if value is None or value < 0:
        value = 0.0
    return value, value


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


# --- 10. Rodar app ---
if __name__ == "__main__":
    app.run(debug=True)