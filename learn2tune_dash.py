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
        # Modelo de nível: A é a área, k é uma constante de saída
        # G(s) = 1 / (As + k)
        A, k = 1.0, 0.5
        return control.TransferFunction([1], [A, k])
    elif tipo == "temperatura":
        # Modelo de temperatura: K é o ganho, tau é a constante de tempo
        # G(s) = K / (tau*s + 1)
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
                            html.H5("Parâmetros do Controlador PID"),
                            html.Hr(),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.Label("Kp (Ganho Proporcional)"),
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
                                            html.Label("Ki (Ganho Integral)"),
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
                                            html.Label("Kd (Ganho Derivativo)"),
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
                        width=4, # Aumentado para melhor visualização dos labels
                    ),
                    # Setpoint e Botões de Controle
                    dbc.Col(
                        [
                            html.H5("Controle e Simulação"),
                            html.Hr(),
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
                            html.Br(),
                            dbc.Button("Simular Distúrbio", id="disturb-button", color="info", className="mt-4"),
                        ],
                        width=4, # Ajustado para acomodar
                    ),
                ],
                className="g-4",
            ),
            dbc.Row(
                dbc.Col(
                    dcc.Graph(
                        id="grafico",
                        style={"width": "100%", "height": "600px"}, # Largura ajustada
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
    return create_main_container("nível")


# --- 7. Callback de simulação ---
@app.callback(
    Output("sim_state", "data"),
    Output("intervalo", "disabled"),
    Input("start", "n_clicks"),
    Input("stop", "n_clicks"),
    Input("reset", "n_clicks"),
    Input("disturb-button", "n_clicks"),
    Input("intervalo", "n_intervals"),
    State("sim_state", "data"),
    State("kp-slider", "value"),
    State("ki-slider", "value"),
    State("kd-slider", "value"),
    State("setpoint", "value"),
    State("url", "pathname"),
)
def atualizar_sim(
    n_start, n_stop, n_reset, n_disturb, n_intervals, sim_state, kp, ki, kd, setpoint, pathname
):
    dt = 0.1
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

    tipo = "temperatura" if pathname == "/temperatura" else "nível"
    
    # Valores padrões para evitar erro de NoneType
    kp_val = kp if kp is not None else 0.0
    ki_val = ki if ki is not None else 0.0
    kd_val = kd if kd is not None else 0.0
    
    # Distúrbio será um offset temporário na variável de processo
    disturbance_offset = 10.0 # Exemplo: aumenta o nível em 10, ou a temperatura em 10°C

    # Reinicializa o estado se for reset ou mudar de página
    if sim_state is None or trigger_id == "reset" or tipo != sim_state.get("tipo"):
        plant_ss = control.ss(criar_planta(tipo))
        X0_plant = np.zeros(plant_ss.nstates)
        y0 = 50.0 if tipo == "nível" else 0.0 # Estado inicial
        
        # Garante que o setpoint inicial esteja no estado da simulação
        initial_sp = setpoint if setpoint is not None else (50.0 if tipo == "nível" else 70.0)

        new_state = {
            "running": False,
            "time": [0],
            "y": [y0],
            "sp": [initial_sp], # Usa o setpoint inicial
            "u": [0.0],
            "X0_plant": X0_plant.tolist(),
            "tipo": tipo,
            "last_time": 0,
            "I": 0.0,
            "prev_error": 0.0,
            "apply_disturbance_at_next_step": False, # Flag para aplicar distúrbio
        }
        return new_state, True

    # Ativa o flag para aplicar o distúrbio no próximo passo de simulação
    if trigger_id == "disturb-button":
        sim_state["apply_disturbance_at_next_step"] = True
    
    if trigger_id == "start":
        sim_state["running"] = True
        return sim_state, False
    if trigger_id == "stop":
        sim_state["running"] = False
        return sim_state, True

    if sim_state["running"]:
        plant_ss = control.ss(criar_planta(tipo))
        X0_plant = np.array(sim_state["X0_plant"])
        y_true_previous = sim_state["y"][-1] # Saída real da planta no passo anterior
        t0 = sim_state["last_time"]
        
        # --- Cálculo da Ação de Controle (u) ---
        # Primeiro, calcula a saída da planta baseada na última ação de controle (u_previous)
        # e no estado anterior X0_plant.
        # Não adicionamos o distúrbio ainda aqui, a planta reage ao 'u' sem distúrbios diretos na sua entrada
        
        # Para simular corretamente, precisamos simular a planta com a entrada 'u'
        # e então adicionar o distúrbio ao resultado 'y' antes de adicionar ruído do sensor.
        
        # AÇÃO DE CONTROLE REALISTA: O PID calcula 'u' com base na 'y_medido' (que contém ruído e possivelmente distúrbio)
        
        # 1. Obter a saída medida (com ruído e distúrbio)
        # 2. Calcular o erro com base na saída medida
        # 3. Calcular a ação de controle 'u'
        # 4. Simular a planta com 'u' como entrada
        # 5. Adicionar o distúrbio à saída 'y' da planta ANTES do ruído do sensor.
        
        # Vamos manter o cálculo de u aqui e aplicar o distúrbio na 'y' no passo da planta

        # Simula a planta com a ação de controle (u_previous) para obter o y_true para este passo
        # Precisamos de um 'u' para a planta. O 'u' é a saída do PID.
        # O PID reage à 'y_medido'. Então, vamos pegar a última 'y_medido' para calcular o 'u' atual.
        # No entanto, 'y_medido' não está salvo no sim_state como histórico, apenas 'y' (o verdadeiro).
        # Para um PID "online", o erro é baseado na leitura atual do sensor.

        # Refazendo a lógica para incorporar o distúrbio *na variável de processo* antes da medição.
        # A saída 'y' da planta (sem ruído ou distúrbio) é calculada primeiro.

        # --- Etapa 1: Simular a planta com a ação de controle anterior ---
        # A planta recebe 'u' e produz 'y'. Vamos usar a última 'u' para o input_output_response
        # ou, mais precisamente, o 'u' que o PID *teria calculado* no passo anterior.
        # Se 'control.input_output_response' pudesse simular para um único passo com uma entrada constante, seria ideal.
        # Como estamos simulando passo a passo, a planta evolui do estado X0_plant usando a última 'u' calculada.

        # Pega a ação de controle atual do histórico (u é a entrada da planta)
        u_plant_input = sim_state["u"][-1] if sim_state["u"] else 0.0

        t_step = [t0, t0 + dt]
        _, y_array_plant_output, X_plant_out = control.input_output_response(
            plant_ss, t_step, [u_plant_input], X0_plant, return_x=True
        )
        y_true_current = float(y_array_plant_output[-1]) # Saída *real* da planta

        # --- Etapa 2: Aplicar distúrbio à saída real da planta ---
        if sim_state["apply_disturbance_at_next_step"]:
            y_true_current += disturbance_offset
            sim_state["apply_disturbance_at_next_step"] = False # Reseta o flag

        # --- Etapa 3: Adicionar ruído de medição para obter 'y_medido' (o que o sensor lê) ---
        if tipo == "nível":
            noise_std = 0.2
        else: # temperatura
            noise_std = 0.5
            
        y_medido = y_true_current + np.random.normal(loc=0, scale=noise_std)
        
        # --- Etapa 4: Calcular o erro e a nova ação de controle 'u' do PID ---
        error = setpoint - y_medido

        sim_state["I"] += error * dt
        D = (error - sim_state["prev_error"]) / dt # Derivada do erro
        
        u_pid_output = kp_val * error + ki_val * sim_state["I"] + kd_val * D
        
        u_pid_output = np.clip(u_pid_output, 0, 100) # Satura a saída do PID
        sim_state["prev_error"] = error

        # --- Etapa 5: Atualizar o estado da simulação ---
        sim_state["time"].append(t0 + dt)
        sim_state["y"].append(y_true_current) # Salva a saída real da planta (com distúrbio)
        sim_state["sp"].append(setpoint)
        sim_state["u"].append(u_pid_output) # Salva a ação de controle calculada pelo PID
        sim_state["X0_plant"] = X_plant_out[:, -1].tolist() # Atualiza o estado da planta
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

    window_duration = 60.0
    current_time = time[-1]
    start_time = max(0, current_time - window_duration)

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
            name="Ação de Controle [%]",
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