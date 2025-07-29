from flask import Flask, render_template, request, send_from_directory, send_file
import os
import pandas as pd
import numpy as np
import plotly
import plotly.graph_objs as go
import json
from scipy import stats
from scipy.optimize import curve_fit


app = Flask(__name__)

S = []



@app.route('/')
def home():

    return render_template('home.html')

@app.route('/aniso_model')
def aniso_model():
    return render_template('aniso_model.html')

@app.route('/iso_model')
def iso_model():  # Mudança feita aqui
    return render_template('iso_model.html')

@app.route('/hellp')
def Hellp():  # Mudança feita aqui
    return render_template('hellp.html')

@app.route('/input_file')
def input_file():  # Mudança feita aqui
    return render_template('input_file.html')

@app.route('/download/instrumental_data')
def download_instrumental_data():
    path_to_csv = 'Instrumental_data.csv'  # Atualize com o caminho correto
    return send_file(path_to_csv, as_attachment=True)

@app.route('/download/experimental_data')
def download_experimental_data():
    path_to_csv = 'Experimental_data.csv'  # Atualize com o caminho correto
    return send_file(path_to_csv, as_attachment=True)


@app.route('/Fit_inst', methods=['GET', 'POST'])
def export_instrumental_file():
    TTH_instru = []
    FWHM_ins = []
    largura_instrumental = []
    Fit_ins = []

    FWHM1 = []
    FWHM = []
    FWHM3 = []

    if request.method == 'POST':
        file_format = request.form['format']

        # Dummy data for the sake of this example
        Lista1 = TTH_instru
        Lista2 = FWHM_ins
        Lista3 = largura_instrumental
        Lista4 = Fit_ins

        # Definindo o cabeçalho
        header = 'TTH_instru FWHM_ins largura_instrumental Fit_ins'
        csv_header = 'TTH_instru,FWHM_ins,largura_instrumental,Fit_ins'

        if file_format == 'txt':
            file_name = "instrumental_data.txt"
            # Adicionando o cabeçalho ao arquivo .txt
            np.savetxt(file_name, np.c_[Lista1, Lista2, Lista3, Lista4], header=header, comments='')
        elif file_format == 'csv':
            file_name = "instrumental_data.csv"
            # Adicionando o cabeçalho ao arquivo .csv
            np.savetxt(file_name, np.c_[Lista1, Lista2, Lista3, Lista4], delimiter=',', header=csv_header, comments='')

        # This will send the file for download
        return send_from_directory(os.path.dirname(os.path.abspath(__file__)), file_name, as_attachment=True)
    # If it's not a POST request, you might want to return something else, like a form to submit the POST request.

@app.route('/', methods=['GET', 'POST'])
def export_experimental_file():
    TTH_instru = []
    FWHM_ins = []
    largura_instrumental = []
    Fit_ins = []

    FWHM1 = []
    FWHM = []
    FWHM3 = []

    if request.method == 'POST':
        file_format = request.form['format']

        # Dummy data for the sake of this example
        Lista1 = TTH_instru
        Lista2 = FWHM_ins
        Lista3 = largura_instrumental
        Lista4 = Fit_ins

        if file_format == 'txt':
            file_name = "experimental_data.txt"
            np.savetxt(file_name, np.c_[Lista1, Lista2, Lista3, Lista4])
        elif file_format == 'csv':
            file_name = "experimental_data.csv"
            np.savetxt(file_name, np.c_[Lista1, Lista2, Lista3, Lista4], delimiter=',')

        # This will send the file for download
        return send_from_directory(os.path.dirname(os.path.abspath(__file__)), file_name, as_attachment=True)
    # If it's not a POST request, you might want to return something else, like a form to submit the POST request.

@app.route('/input_form', methods=['GET', 'POST'])
def input_form1():
    if request.method == 'POST':
        for i in range(1, 37):
            value = request.form.get(f'float_{i}')
            if value:
                S.append(float(value))
    return render_template('input_form.html')

@app.route('/import/instrumental-file', methods=['POST'])
def import_instrumental_file():
    TTH_instru = []
    FWHM_ins = []
    largura_instrumental = []
    Fit_ins = []

    FWHM1 = []
    FWHM = []
    FWHM3 = []

    popt = []
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        df = pd.read_csv(uploaded_file)
        x = TH = df["2-theta"] * (np.pi / 360)  # Usar TH
        y = FWHM = df["FWHM"]
        for i in x:
            TTH_instru.append(i)

        for i in y:
            FWHM_ins.append(i)

        def func(x, U, V, W, X, Y):  # Largura total
            gama_g = ((8 * 0.693) ** 0.5) * ((U * np.tan(x) ** 2) + V * np.tan(x) + W)
            Lorenzian = X / (np.tan(x)) + Y * (np.cos(x))
            return (gama_g ** 5 + 2.6926 * (gama_g ** 4) * Lorenzian + 2.42843 * (
                        gama_g ** 3) * Lorenzian ** 2 + 4.47163 * (gama_g ** 2) * Lorenzian ** 3 + 0.07842 * (
                        gama_g) * Lorenzian ** 4 + Lorenzian ** 5) ** (1 / 5)

        popt2, pcov2 = curve_fit(func, x, y)
        U = popt2[0]
        V = popt2[1]
        W = popt2[2]
        X = popt2[3]
        Y = popt2[4]

        popt.append(U)
        popt.append(V)
        popt.append(W)
        popt.append(X)
        popt.append(Y)

        def Loren(x):  # largura_total_ajuste
            gama_g = ((8 * 0.693) ** 0.5) * (popt2[0] * np.tan(x) ** 2 + popt2[1] * np.tan(x) + popt2[2])
            Lorenzian = popt2[3] / (np.tan(x)) + popt2[4] * (np.cos(x))

            return (gama_g ** 5 + 2.6926 * (gama_g ** 4) * Lorenzian + 2.42843 * (
                        gama_g ** 3) * Lorenzian ** 2 + 4.47163 * (gama_g ** 2) * Lorenzian ** 3 + 0.07842 * (
                        gama_g) * Lorenzian ** 4 + Lorenzian ** 5) ** (1 / 5)

        for i in x:
            Fit_ins.append(Loren(i))

        for i in TH:
            largura_ins = Loren(i)
            largura_instrumental.append(largura_ins)

        trace1 = go.Scatter(x=x, y=y, mode='markers', name='EXP')
        trace2 = go.Scatter(x=x, y=Loren(x), mode='lines', name='Ajuste', line=dict(color='red'))

        layout = go.Layout(
            title={
                'text': f'Fit - Instrumental File',
                'font': {
                    'size': 24,  # Ajuste o tamanho da fonte conforme necessário
                    'color': 'black',  # A cor do texto do título
                    'family': 'Arial Bold, sans-serif',  # A fonte usada para o título em negrito
                },
                'x': 0.5,  # Centraliza o título no eixo x
                'y': 0.95,  # Ajusta a posição vertical do título, 0.95 é quase no topo
                'xanchor': 'center',  # Alinha o centro do título com a posição 'x'
                'yanchor': 'top'  # Alinha a parte superior do título com a posição 'y'
            },
            xaxis=dict(title='2θ (rad)'),
            yaxis=dict(title='FWHM(°)'),
            showlegend=True
        )

        figura = go.Figure(data=[trace1, trace2], layout=layout)
        graphJSON = json.dumps(figura, cls=plotly.utils.PlotlyJSONEncoder)
        return render_template('Fit_inst.html', graphJSON=graphJSON)


@app.route('/plot', methods=['POST'])
def plot():
    TTH_instru = []
    FWHM_ins = []
    largura_instrumental = []
    Fit_ins = []

    FWHM1 = []
    FWHM = []
    FWHM3 = []

    TH = []
    TH1 = []

    Dm_WHP = []
    e_WHP = []
    yr1 = []
    WHP_x = []
    WHP_y = []
    Sch = []

    Dm_SSP = []
    e_SSP = []
    yr2 = []
    SSP_x = []
    SSP_y = []
    d = []

    Dm_HWP = []
    e_HWP = []
    yr3 = []
    HWP_x = []
    HWP_y = []

    wavelengt = float()
    formfactor = float()
    wave_length = request.form.get("wavelength")
    form_factor = request.form.get("formfactor")
    formfactor = float(form_factor)
    print(formfactor)
    modelo_selecionado = request.form.get('selecao_modelo')
    uploaded_file = request.files['file']
    if wave_length == "Cu":
        wavelengt = 0.154056
    elif wave_length == "Mo":
        wavelengt = 0.070930
    elif wave_length == "Co":
        wavelengt = 0.178897
    elif wave_length == "Cr":
        wavelengt = 1
    elif wave_length == "Fe":
        wavelengt = 0.193604
    elif wave_length == "Ag":
        wavelengt = 0.022941
    else:
        pass

    if uploaded_file.filename != '':
        df = pd.read_csv(uploaded_file)
        FWHM2 = df["FWHM"]
        TTH = df["2-theta"]
        TH.append(TTH)
        for i in FWHM2:
            FWHM1.append(i)

        for i in range(len(FWHM1)):
            if i < len(largura_instrumental):
                FWHM.append(FWHM1[i] - largura_instrumental[i])
            else:
                FWHM.append(0)
                TH.append(0)

        if modelo_selecionado == "WHP":

            for i in range(len(TTH)):

                if FWHM[i] > 0:
                    Sch1 = wavelengt / (FWHM[i] * np.pi / 180) / np.cos(TTH[i] * np.pi / 360)
                    WHP_x1 = 4 * np.sin(TTH[i] * np.pi / 360) / wavelengt
                    WHP_y1 = 1 / (Sch1)
                    Sch.append(Sch1)
                    WHP_x.append(WHP_x1)
                    WHP_y.append(WHP_y1)

                else:
                    Sch1 = wavelengt / (FWHM1[i] * np.pi / 180) / np.cos(TTH[i] * np.pi / 360)
                    WHP_x1 = 4 * np.sin(TTH[i] * np.pi / 360) / wavelengt
                    WHP_y1 = 1 / (Sch1)
                    Sch.append(Sch1)
                    WHP_x.append(WHP_x1)
                    WHP_y.append(WHP_y1)

            valid_indices = [i for i in range(len(WHP_x)) if not (np.isnan(WHP_x[i]) or np.isnan(WHP_y[i]))]

            # Filtra os valores válidos de WHP_x e WHP_y
            WHP_x_filtered = [WHP_x[i] for i in valid_indices]
            WHP_y_filtered = [WHP_y[i] for i in valid_indices]

            # Realiza o ajuste linear com os valores filtrados
            p1 = np.polyfit(WHP_x_filtered, WHP_y_filtered, 1)

            yr = np.polyval(p1, WHP_x_filtered)  # Ajuste Linear

            for i in yr:
                yr1.append(i)

            R1 = stats.linregress(WHP_x_filtered, WHP_y_filtered)  # Parâmetros de confiança
            R1_2 = R1.rvalue ** 2  # Parâmetros de confiança
            # R1_2.append(R1_21)
            Dm_WHP1 = formfactor / p1[1]  # Tamanho
            Dm_WHP.append(Dm_WHP1)
            print(Dm_WHP1)
            e_WHP1 = p1[0]
            e_WHP.append(e_WHP1)

            # Dados e ajuste
            trace1 = go.Scatter(x=WHP_x_filtered, y=WHP_y_filtered, mode='markers', name='WHP')
            trace2 = go.Scatter(x=WHP_x_filtered, y=yr1, mode='lines', name='Ajuste', line=dict(color='red'))

            # Criar a figura e adicionar os traços

            # Título e labels dos eixos
            title_text = r'$<D> = {:.2f} nm     \ \ \ \   \varepsilon = {:.2e}      \ \ \ \        R^2 = {:.4f}$'.format(
                Dm_WHP1, e_WHP1, R1_2)
            fig = go.Figure(data=[trace1, trace2])
            fig.update_layout(title=title_text, xaxis_title=r'$ \sin{\theta}$',
                              yaxis_title=r'$ \beta * \cos {\theta} / \lambda $')
            graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            return render_template('plot.html', graphJSON=graphJSON)

        if modelo_selecionado == "SSP":
            for i in range(len(TTH)):

                if FWHM[i] > 0:
                    d1 = wavelengt / 2 / np.sin(TTH[i] * np.pi / 360)
                    SSP_x1 = d1 ** 2
                    SSP_y1 = (d1 * (FWHM[i] * np.pi / 180) * np.cos(TTH[i] * np.pi / 360) / wavelengt) ** 2
                    SSP_x.append(SSP_x1)
                    SSP_y.append(SSP_y1)

                else:
                    d1 = wavelengt / 2 / np.sin(TTH[i] * np.pi / 360)
                    SSP_x1 = d1 ** 2
                    SSP_y1 = (d1 * (FWHM1[i] * np.pi / 180) * np.cos(TTH[i] * np.pi / 360) / wavelengt) ** 2
                    SSP_x.append(SSP_x1)
                    SSP_y.append(SSP_y1)

            valid_indices = [i for i in range(len(SSP_x)) if not (np.isnan(SSP_x[i]) or np.isnan(SSP_y[i]))]

            # Filtra os valores válidos de SSP_x e SSP_y
            SSP_x_filtered = [SSP_x[i] for i in valid_indices]
            SSP_y_filtered = [SSP_y[i] for i in valid_indices]

            # Realiza o ajuste linear com os valores filtrados
            p2 = np.polyfit(SSP_x_filtered, SSP_y_filtered, 1)

            yr = np.polyval(p2, SSP_x_filtered)  # Ajuste Linear
            for i in yr:
                yr2.append(i)
            R2 = stats.linregress(SSP_x_filtered, SSP_y_filtered)  # Parâmetros de confiança
            R2_2 = R2.rvalue ** 2  # Parâmetros de confiança
            Dm_SSP1 = (formfactor) / np.sqrt(p2[0])  # Tamanho
            e_SSP1 = np.sqrt(p2[1] / 2 * np.pi)

            Dm_SSP.append(Dm_SSP1)
            e_SSP.append(e_SSP1)

            # Dados e ajuste
            trace1 = go.Scatter(x=SSP_x_filtered, y=SSP_y_filtered, mode='markers', name='SSP')
            trace2 = go.Scatter(x=SSP_x_filtered, y=yr2, mode='lines', name='Ajuste', line=dict(color='red'))

            # Criar a figura e adicionar os traços

            # Título e labels dos eixos
            title_text = r'$<D> = {:.2f} nm     \ \ \ \   \varepsilon = {:.2e}      \ \ \ \        R^2 = {:.4f}$'.format(
                Dm_SSP1, e_SSP1, R2_2)
            fig = go.Figure(data=[trace1, trace2])
            fig.update_layout(title=title_text, xaxis_title=r'$d^2$',
                              yaxis_title=r'$(d^2 \beta * \cos{\theta} / \lambda )^2$')
            graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            return render_template('plot.html', graphJSON=graphJSON)

        if modelo_selecionado == "HWP":
            for i in range(len(TTH)):

                if FWHM[i] > 0:
                    Sch1 = wavelengt/ (FWHM[i] * np.pi / 180) / np.cos(TTH[i] * np.pi / 360)
                    d1 = wavelengt / 2 / np.sin(TTH[i] * np.pi / 360)
                    b1 = 1 / Sch1
                    d2 = 1 / d1
                    HWP_x1 = b1 / (d2) ** 2
                    HWP_y1 = (b1 / d2) ** 2

                    HWP_x.append(HWP_x1)
                    HWP_y.append(HWP_y1)
                else:
                    Sch1 = wavelengt / (FWHM1[i] * np.pi / 180) / np.cos(TTH[i] * np.pi / 360)
                    d1 = wavelengt / 2 / np.sin(TTH[i] * np.pi / 360)
                    b1 = 1 / Sch1
                    d2 = 1 / d1
                    HWP_x1 = b1 / (d2) ** 2
                    HWP_y1 = (b1 / d2) ** 2

                    HWP_x.append(HWP_x1)
                    HWP_y.append(HWP_y1)

            valid_indices = [i for i in range(len(HWP_x)) if not (np.isnan(HWP_x[i]) or np.isnan(HWP_y[i]))]

            # Filtra os valores válidos de HWP_x e HWP_y
            HWP_x_filtered = [HWP_x[i] for i in valid_indices]
            HWP_y_filtered = [HWP_y[i] for i in valid_indices]

            # Realiza o ajuste linear com os valores filtrados
            p1 = np.polyfit(HWP_x_filtered, HWP_y_filtered, 1)
            yr = np.polyval(p1, HWP_x_filtered)  # Ajuste Linear
            for i in yr:
                yr3.append(i)

            # yr1.append(yr)
            R1 = stats.linregress(HWP_x_filtered, HWP_y_filtered)  # Parâmetros de confiança
            R3_2 = R1.rvalue ** 2  # Parâmetros de confiança
            # R1_2.append(R1_21)
            Dm_HWP1 = formfactor / p1[0]  # Tamanho
            Dm_HWP.append(Dm_HWP1)
            e_HWP1 = p1[1]
            e_HWP.append(e_HWP1)

            # Dados e ajuste
            trace1 = go.Scatter(x=HWP_x_filtered, y=HWP_y_filtered, mode='markers', name='HWP')
            trace2 = go.Scatter(x=HWP_x_filtered, y=yr3, mode='lines', name='Ajuste', line=dict(color='red'))

            # Criar a figura e adicionar os traços

            # Título e labels dos eixos
            title_text = r'$<D> = {:.2f} nm     \ \ \ \   \varepsilon = {:.2e}      \ \ \ \        R^2 = {:.4f}$'.format(
                Dm_HWP1, e_HWP1, R3_2)
            fig = go.Figure(data=[trace1, trace2])
            fig.update_layout(title=title_text, xaxis_title=r'$ \beta * \cos  / d*^2$',
                              yaxis_title=r'$ (\beta * / d*)^2$')
            graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            return render_template('plot.html', graphJSON=graphJSON)


@app.route('/plot_ani', methods=['POST'])
def plot_ani():
    TTH_instru = []
    FWHM_ins = []
    largura_instrumental = []
    Fit_ins = []

    FWHM1 = []
    FWHM = []
    FWHM3 = []

    TH = []
    TH1 = []

    h1 = []
    l1 = []
    k1 = []

    Ehkl_1 = []
    Ehkl = []

    e_UDSM = []
    yr4 = []
    UDSM_x = []
    UDSM_y = []

    e_UDEDM = []
    yr5 = []
    UDEDM_x = []
    UDEDM_y = []

    e_SSP_UDSM = []
    yr6 = []
    SSP_UDSM_x = []
    SSP_UDSM_y = []

    wavelengt = float()

    wave_length = request.form.get("wavelength")
    form_factor = request.form.get("formfactor")
    formfactor = float(form_factor)
    modelo_selecionado = request.form.get('selecao_modelo')
    uploaded_file = request.files['file']
    if wave_length == "Cu":
        wavelengt = 0.154056
    elif wave_length == "Mo":
        wavelengt = 0.070930
    elif wave_length == "Co":
        wavelengt = 0.178897
    elif wave_length == "Cr":
        wavelengt = 1
    elif wave_length == "Fe":
        wavelengt = 0.193604
    elif wave_length == "Ag":
        wavelengt = 0.022941
    else:
        pass

    if uploaded_file.filename != '':
        df = pd.read_csv(uploaded_file)
        TTH = df["2-theta"]
        FWHM2 = df["FWHM"]
        for i in range(len(FWHM2)):
            if i < len(largura_instrumental):
                FWHM3.append(FWHM2[i] - largura_instrumental[i])
                TH1.append(TTH[i])
            else:
                FWHM3.append(0)
                TH1.append(0)
        h = df["h"]
        k = df["k"]
        l = df["l"]

        for i, j, m in zip(h, k, l):
            h1.append(i)
            k1.append(j)
            l1.append(m)

        if modelo_selecionado == "UDSM":
            for i in range(len(TTH)):

                s_11 = S[0]
                s_12 = S[1]
                s_13 = S[2]
                s_14 = S[3]
                s_15 = S[4]
                s_16 = S[5]

                s_21 = S[6]
                s_22 = S[7]
                s_23 = S[8]
                s_24 = S[9]
                s_25 = S[10]
                s_26 = S[11]

                s_31 = S[12]
                s_32 = S[13]
                s_33 = S[14]
                s_34 = S[15]
                s_35 = S[16]
                s_36 = S[17]

                s_41 = S[18]
                s_42 = S[19]
                s_43 = S[20]
                s_44 = S[21]
                s_45 = S[22]
                s_46 = S[23]

                s_51 = S[24]
                s_52 = S[25]
                s_53 = S[26]
                s_54 = S[27]
                s_55 = S[28]
                s_56 = S[29]

                s_61 = S[30]
                s_62 = S[31]
                s_63 = S[32]
                s_64 = S[33]
                s_65 = S[34]
                s_66 = S[35]

                # Cubic System - All Classes
                if s_11 != 0 and s_12 != 0 and s_44 != 0:
                    E_hkl1 = s_11 - 2 * (s_11 - s_12 - s_44 / 2) * (
                    (h[i] ** 2 * k[i] ** 2 + h[i] ** 2 * l[i] ** 2 + k[i] ** 2 * l[i] ** 2)) / (
                                         h[i] ** 2 + k[i] ** 2 + l[i] ** 2) ** 2

                    Ehkl_1.append(E_hkl1)
                    Ehkl.append(1 / E_hkl1)
                    # print(1 / E_hkl1)
                    UDSM_x1 = 4 * np.sin(TTH[i] * np.pi / 360) / Ehkl[i] / wavelengt
                    UDSM_y1 = (FWHM2[i] * np.pi / 180) * np.cos(TTH[i] * np.pi / 360) / wavelengt

                    UDSM_x.append(UDSM_x1)
                    UDSM_y.append(UDSM_y1)

                # Orthorhombic System - All Classes
                if s_11 != 0 and s_12 != 0 and s_13 != 0 and s_22 != 0 and s_23 != 0 and s_33 != 0 and s_44 != 0 and s_55 != 0 and s_66 != 0:
                    E_hkl1 = ((h[i] ** 4) * s_11 + 2 * (h[i] ** 2) * (k[i] ** 2) * s_12 + 2 * (h[i] ** 2) * (
                                l[i] ** 2) * s_13 + (k[i] ** 4) * s_22 + 2 * (k[i] ** 2) * (l[i] ** 2) * s_23 + (
                                          l[i] ** 4) * s_33 + (k[i] ** 2) * (l[i] ** 2) * s_44 + (h[i] ** 2) * (
                                          l[i] ** 2) * s_55 + (h[i] ** 2) * (l[i] ** 2) * s_66) * (
                             (h[i] ** 2 * k[i] ** 2 + h[i] ** 2 * l[i] ** 2 + k[i] ** 2 * l[i] ** 2)) / (
                                         h[i] ** 2 + k[i] ** 2 + l[i] ** 2) ** 2

                    Ehkl_1.append(E_hkl1)
                    Ehkl.append(1 / E_hkl1)
                    # print(1 / E_hkl1)
                    UDSM_x1 = 4 * np.sin(TTH[i] * np.pi / 360) / Ehkl[i] / wavelengt
                    UDSM_y1 = (FWHM2[i] * np.pi / 180) * np.cos(TTH[i] * np.pi / 360) / wavelengt

                    UDSM_x.append(UDSM_x1)
                    UDSM_y.append(UDSM_y1)

                # Tetragonal System - Classes 4, -4, 4/m
                if s_11 != 0 and s_12 != 0 and s_13 != 0 and s_33 != 0 and s_44 != 0 and s_16 != 0 and s_66 != 0:
                    E_hkl1 = ((h[i] ** 4 + k[i] ** 4) * s_11 + (l[i] ** 4) * s_33 + (h[i] ** 2) * (k[i] ** 2) * (
                                2 * s_12 + s_66) + (l[i] ** 2) * (1 - l[i] ** 2) * (2 * s_13 + s_14) + (
                                          2 * h[i] * k[i]) * (h[i] ** 2 - k[i] ** 2) * s_16) * (
                             (h[i] ** 2 * k[i] ** 2 + h[i] ** 2 * l[i] ** 2 + k[i] ** 2 * l[i] ** 2)) / (
                                         h[i] ** 2 + k[i] ** 2 + l[i] ** 2) ** 2

                    Ehkl_1.append(E_hkl1)
                    Ehkl.append(1 / E_hkl1)
                    # print(1 / E_hkl1)
                    UDSM_x1 = 4 * np.sin(TTH[i] * np.pi / 360) / Ehkl[i] / wavelengt
                    UDSM_y1 = (FWHM2[i] * np.pi / 180) * np.cos(TTH[i] * np.pi / 360) / wavelengt

                    UDSM_x.append(UDSM_x1)
                    UDSM_y.append(UDSM_y1)

                # Trigonal System - Classes 3, -3
                if s_11 != 0 and s_13 != 0 and s_33 != 0 and s_44 != 0 and s_14 != 0 and s_25 != 0:
                    E_hkl1 = ((1 - l[i] ** 2) ** 2 * s_11 + (l[i] ** 4) * s_33 + (l[i] ** 2) * (1 - l[i] ** 2) * (
                                2 * s_13 + s_44) + (2 * k[i] * l[i]) * (3 * h[i] ** 2 - k[i] ** 2) * s_14 + (
                                          2 * h[i] * l[i]) * (3 * k[i] ** 2 - h[i] ** 2) * s_25) * (
                             (h[i] ** 2 * k[i] ** 2 + h[i] ** 2 * l[i] ** 2 + k[i] ** 2 * l[i] ** 2)) / (
                                         h[i] ** 2 + k[i] ** 2 + l[i] ** 2) ** 2

                    Ehkl_1.append(E_hkl1)
                    Ehkl.append(1 / E_hkl1)
                    # print(1 / E_hkl1)
                    UDSM_x1 = 4 * np.sin(TTH[i] * np.pi / 360) / Ehkl[i] / wavelengt
                    UDSM_y1 = (FWHM2[i] * np.pi / 180) * np.cos(TTH[i] * np.pi / 360) / wavelengt

                    UDEDM_x.append(UDSM_x1)
                    UDEDM_y.append(UDSM_y1)

                # Hexagonal System - All Classes
                if s_11 != 0 and s_13 != 0 and s_33 != 0 and s_44 != 0:
                    E_hkl1 = ((1 - l[i] ** 2) ** 2 * s_11 + (l[i] ** 4) * s_33 + (l[i] ** 2) * (1 - l[i] ** 2) * (
                                2 * s_13 + s_44)) * (
                                         h[i] ** 2 * k[i] ** 2 + h[i] ** 2 * l[i] ** 2 + k[i] ** 2 * l[i] ** 2) / (
                                         h[i] ** 2 + k[i] ** 2 + l[i] ** 2) ** 2
                    Ehkl_1.append(E_hkl1)
                    Ehkl.append(1 / E_hkl1)
                    UDSM_x1 = 4 * np.sin(TTH[i] * np.pi / 360) / Ehkl[i] / wavelengt
                    UDSM_y1 = (FWHM2[i] * np.pi / 180) * np.cos(TTH[i] * np.pi / 360) / wavelengt
                    UDSM_x.append(UDSM_x1)
                    UDSM_y.append(UDSM_y1)

                # Triclinic System - All Classes
                if s_11 != 0 and s_12 != 0 and s_13 != 0 and s_14 != 0 and s_15 != 0 and s_16 != 0 and s_21 != 0 and s_22 != 0 and s_23 != 0 and s_24 != 0 and s_25 != 0 and s_26 != 0 and s_33 != 0 and s_44 != 0 and s_55 != 0 and s_66 != 0:
                    E_hkl1 = ((h[i] ** 4) * s_11 + 2 * (h[i] ** 2) * (k[i] ** 2) * s_12 + 2 * (h[i] ** 2) * (
                                l[i] ** 2) * s_13 + 2 * (h[i] ** 2) * (k[i] * l[i]) * s_14 + 2 * (h[i] ** 3) * (
                              l[i]) * s_15 + 2 * (h[i] ** 3) * (k[i]) * s_16 + (k[i] ** 4) * s_22 + 2 * (k[i] ** 2) * (
                                          l[i] ** 2) * s_23 + 2 * (k[i] ** 3) * (l[i]) * s_24 + 2 * (h[i]) * (
                                          k[i] ** 2) * (l[i]) * s_25 + 2 * (h[i]) * (k[i] ** 3) * s_26 + (
                                          l[i] ** 4) * s_33 + 2 * (k[i]) * (l[i] ** 3) * s_34 + 2 * (h[i]) * (
                                          l[i] ** 3) * s_35 + 2 * (h[i]) * (l[i] ** 2) * (k[i]) * s_36 + (k[i] ** 2) * (
                                          l[i] ** 2) * s_44 + 2 * (h[i]) * (k[i]) * (l[i] ** 2) * s_45 + 2 * (h[i]) * (
                                          k[i] ** 2) * (l[i]) * s_46 + (h[i] ** 2) * (l[i] ** 2) * s_55 + 2 * (
                                          h[i] ** 2) * (k[i]) * (l[i]) * s_56 + (h[i] ** 2) * (k[i] ** 2) * s_66) * (
                                         h[i] ** 2 * k[i] ** 2 + h[i] ** 2 * l[i] ** 2 + k[i] ** 2 * l[i] ** 2) / (
                                         h[i] ** 2 + k[i] ** 2 + l[i] ** 2) ** 2
                    Ehkl_1.append(E_hkl1)
                    Ehkl.append(1 / E_hkl1)
                    # print(1 / E_hkl1)
                    UDSM_x1 = 4 * np.sin(TTH[i] * np.pi / 360) / Ehkl[i] / wavelengt
                    UDSM_y1 = (FWHM2[i] * np.pi / 180) * np.cos(TTH[i] * np.pi / 360) / wavelengt

                    UDEDM_x.append(UDSM_x1)
                    UDEDM_y.append(UDSM_y1)

            valid_indices = [i for i in range(len(UDSM_x)) if not (np.isnan(UDSM_x[i]) or np.isnan(UDSM_y[i]))]

            # Filtra os valores válidos de UDEDM_x e UDEDM_Y
            UDSM_x_filtered = [UDSM_x[i] for i in valid_indices]
            UDSM_y_filtered = [UDSM_y[i] for i in valid_indices]

            # Realiza o ajuste linear com os valores filtrados
            p5 = np.polyfit(UDSM_x_filtered, UDSM_y_filtered, 1)
            # print(p5)

            Dm = formfactor / p5[1]  # Tamanho

            # e_UDSM = []

            for i in Ehkl:
                e1 = np.sqrt(2 / i) * p5[0]
                e_UDSM.append(e1)

            e_media = sum(e_UDSM) / len(e_UDSM)

            yr = np.polyval(p5, UDSM_x_filtered)  # Ajuste Linear
            for i in yr:
                yr4.append(i)
            R4 = stats.linregress(UDSM_x_filtered, UDSM_y_filtered)  # Parâmetros de confiança
            R4_2 = R4.rvalue ** 2  # Parâmetros de confiança

            # Dados e ajuste
            trace1 = go.Scatter(x=UDSM_x_filtered, y=UDSM_y_filtered, mode='markers', name='HWP - UDSM ')
            trace2 = go.Scatter(x=UDSM_x_filtered, y=yr4, mode='lines', name='Ajuste', line=dict(color='red'))

            # Criar a figura e adicionar os traços
            title_text = r'$<D> = {:.2f} nm     \ \ \ \   \varepsilon = {:.2e}      \ \ \ \        R^2 = {:.4f}$'.format(
                Dm, e_media, R4_2)
            fig = go.Figure(data=[trace1, trace2])
            fig.update_layout(title=title_text, xaxis_title=r'$ 4 \sin (\theta) / \lambda E_{hkl} $',
                              yaxis_title=r'$ \beta \cos (\theta) / \lambda $')
            graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

            # Título e labels dos eixos
            title_text = r'$<D> = {:.2f} nm     \ \ \ \   \varepsilon = {:.2e}      \ \ \ \        R^2 = {:.4f}$'.format(
                Dm, e_media, R4_2)
            fig = go.Figure(data=[trace1, trace2])
            fig.update_layout(title=title_text, xaxis_title=r'$ \beta * \cos  / d*^2$',
                              yaxis_title=r'$ (\beta * / d*)^2$')
            graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            return render_template('plot.html', graphJSON=graphJSON)

        if modelo_selecionado == "UDEDM":
            for i in range(len(TTH)):

                s_11 = S[0]
                s_12 = S[1]
                s_13 = S[2]
                s_14 = S[3]
                s_15 = S[4]
                s_16 = S[5]

                s_21 = S[6]
                s_22 = S[7]
                s_23 = S[8]
                s_24 = S[9]
                s_25 = S[10]
                s_26 = S[11]

                s_31 = S[12]
                s_32 = S[13]
                s_33 = S[14]
                s_34 = S[15]
                s_35 = S[16]
                s_36 = S[17]

                s_41 = S[18]
                s_42 = S[19]
                s_43 = S[20]
                s_44 = S[21]
                s_45 = S[22]
                s_46 = S[23]

                s_51 = S[24]
                s_52 = S[25]
                s_53 = S[26]
                s_54 = S[27]
                s_55 = S[28]
                s_56 = S[29]

                s_61 = S[30]
                s_62 = S[31]
                s_63 = S[32]
                s_64 = S[33]
                s_65 = S[34]
                s_66 = S[35]

                # Cubic System - All Classes
                if s_11 != 0 and s_12 != 0 and s_44 != 0:
                    E_hkl1 = s_11 - 2 * (s_11 - s_12 - s_44 / 2) * (
                    (h[i] ** 2 * k[i] ** 2 + h[i] ** 2 * l[i] ** 2 + k[i] ** 2 * l[i] ** 2)) / (
                                         h[i] ** 2 + k[i] ** 2 + l[i] ** 2) ** 2

                    Ehkl_1.append(E_hkl1)
                    Ehkl.append(1 / E_hkl1)
                    # print(1 / E_hkl1)
                    UDEDM_x1 = 5.657 * np.sin(TTH[i] * np.pi / 360) / np.sqrt(1 / E_hkl1) / wavelengt
                    UDEDM_y1 = (FWHM2[i] * np.pi / 180) * np.cos(TTH[i] * np.pi / 360) / wavelengt

                    UDEDM_x.append(UDEDM_x1)
                    UDEDM_y.append(UDEDM_y1)

                # Orthorhombic System - All Classes
                if s_11 != 0 and s_12 != 0 and s_13 != 0 and s_22 != 0 and s_23 != 0 and s_33 != 0 and s_44 != 0 and s_55 != 0 and s_66 != 0:
                    E_hkl1 = ((h[i] ** 4) * s_11 + 2 * (h[i] ** 2) * (k[i] ** 2) * s_12 + 2 * (h[i] ** 2) * (
                                l[i] ** 2) * s_13 + (k[i] ** 4) * s_22 + 2 * (k[i] ** 2) * (l[i] ** 2) * s_23 + (
                                          l[i] ** 4) * s_33 + (k[i] ** 2) * (l[i] ** 2) * s_44 + (h[i] ** 2) * (
                                          l[i] ** 2) * s_55 + (h[i] ** 2) * (l[i] ** 2) * s_66) * (
                             (h[i] ** 2 * k[i] ** 2 + h[i] ** 2 * l[i] ** 2 + k[i] ** 2 * l[i] ** 2)) / (
                                         h[i] ** 2 + k[i] ** 2 + l[i] ** 2) ** 2

                    Ehkl_1.append(E_hkl1)
                    Ehkl.append(1 / E_hkl1)
                    # print(1 / E_hkl1)
                    UDEDM_x1 = 5.657 * np.sin(TTH[i] * np.pi / 360) / np.sqrt(1 / E_hkl1) / wavelengt
                    UDEDM_y1 = (FWHM2[i] * np.pi / 180) * np.cos(TTH[i] * np.pi / 360) / wavelengt

                    UDEDM_x.append(UDEDM_x1)
                    UDEDM_y.append(UDEDM_y1)

                # Tetragonal System - Classes 4, -4, 4/m
                if s_11 != 0 and s_12 != 0 and s_13 != 0 and s_33 != 0 and s_44 != 0 and s_16 != 0 and s_66 != 0:
                    E_hkl1 = ((h[i] ** 4 + k[i] ** 4) * s_11 + (l[i] ** 4) * s_33 + (h[i] ** 2) * (k[i] ** 2) * (
                                2 * s_12 + s_66) + (l[i] ** 2) * (1 - l[i] ** 2) * (2 * s_13 + s_14) + (
                                          2 * h[i] * k[i]) * (h[i] ** 2 - k[i] ** 2) * s_16) * (
                             (h[i] ** 2 * k[i] ** 2 + h[i] ** 2 * l[i] ** 2 + k[i] ** 2 * l[i] ** 2)) / (
                                         h[i] ** 2 + k[i] ** 2 + l[i] ** 2) ** 2

                    Ehkl_1.append(E_hkl1)
                    Ehkl.append(1 / E_hkl1)
                    # print(1 / E_hkl1)
                    UDEDM_x1 = 5.657 * np.sin(TTH[i] * np.pi / 360) / np.sqrt(1 / E_hkl1) / wavelengt
                    UDEDM_y1 = (FWHM2[i] * np.pi / 180) * np.cos(TTH[i] * np.pi / 360) / wavelengt

                    UDEDM_x.append(UDEDM_x1)
                    UDEDM_y.append(UDEDM_y1)

                # Trigonal System - Classes 3, -3
                if s_11 != 0 and s_13 != 0 and s_33 != 0 and s_44 != 0 and s_14 != 0 and s_25 != 0:
                    E_hkl1 = ((1 - l[i] ** 2) ** 2 * s_11 + (l[i] ** 4) * s_33 + (l[i] ** 2) * (1 - l[i] ** 2) * (
                                2 * s_13 + s_44) + (2 * k[i] * l[i]) * (3 * h[i] ** 2 - k[i] ** 2) * s_14 + (
                                          2 * h[i] * l[i]) * (3 * k[i] ** 2 - h[i] ** 2) * s_25) * (
                             (h[i] ** 2 * k[i] ** 2 + h[i] ** 2 * l[i] ** 2 + k[i] ** 2 * l[i] ** 2)) / (
                                         h[i] ** 2 + k[i] ** 2 + l[i] ** 2) ** 2

                    Ehkl_1.append(E_hkl1)
                    Ehkl.append(1 / E_hkl1)
                    # print(1 / E_hkl1)
                    UDEDM_x1 = 5.657 * np.sin(TTH[i] * np.pi / 360) / np.sqrt(1 / E_hkl1) / wavelengt
                    UDEDM_y1 = (FWHM2[i] * np.pi / 180) * np.cos(TTH[i] * np.pi / 360) / wavelengt

                    UDEDM_x.append(UDEDM_x1)
                    UDEDM_y.append(UDEDM_y1)

                # Hexagonal System - All Classes
                if s_11 != 0 and s_13 != 0 and s_33 != 0 and s_44 != 0:
                    E_hkl1 = ((1 - l[i] ** 2) ** 2 * s_11 + (l[i] ** 4) * s_33 + (l[i] ** 2) * (1 - l[i] ** 2) * (
                                2 * s_13 + s_44)) * (
                                         h[i] ** 2 * k[i] ** 2 + h[i] ** 2 * l[i] ** 2 + k[i] ** 2 * l[i] ** 2) / (
                                         h[i] ** 2 + k[i] ** 2 + l[i] ** 2) ** 2
                    Ehkl_1.append(E_hkl1)
                    Ehkl.append(1 / E_hkl1)
                    UDEDM_x1 = 5.657 * np.sin(TTH[i] * np.pi / 360) / np.sqrt(1 / E_hkl1) / wavelengt
                    UDEDM_y1 = (FWHM2[i] * np.pi / 180) * np.cos(TTH[i] * np.pi / 360) / wavelengt
                    UDEDM_x.append(UDEDM_x1)
                    UDEDM_y.append(UDEDM_y1)

                # Triclinic System - All Classes
                if s_11 != 0 and s_12 != 0 and s_13 != 0 and s_14 != 0 and s_15 != 0 and s_16 != 0 and s_21 != 0 and s_22 != 0 and s_23 != 0 and s_24 != 0 and s_25 != 0 and s_26 != 0 and s_33 != 0 and s_44 != 0 and s_55 != 0 and s_66 != 0:
                    E_hkl1 = ((h[i] ** 4) * s_11 + 2 * (h[i] ** 2) * (k[i] ** 2) * s_12 + 2 * (h[i] ** 2) * (
                                l[i] ** 2) * s_13 + 2 * (h[i] ** 2) * (k[i] * l[i]) * s_14 + 2 * (h[i] ** 3) * (
                              l[i]) * s_15 + 2 * (h[i] ** 3) * (k[i]) * s_16 + (k[i] ** 4) * s_22 + 2 * (k[i] ** 2) * (
                                          l[i] ** 2) * s_23 + 2 * (k[i] ** 3) * (l[i]) * s_24 + 2 * (h[i]) * (
                                          k[i] ** 2) * (l[i]) * s_25 + 2 * (h[i]) * (k[i] ** 3) * s_26 + (
                                          l[i] ** 4) * s_33 + 2 * (k[i]) * (l[i] ** 3) * s_34 + 2 * (h[i]) * (
                                          l[i] ** 3) * s_35 + 2 * (h[i]) * (l[i] ** 2) * (k[i]) * s_36 + (k[i] ** 2) * (
                                          l[i] ** 2) * s_44 + 2 * (h[i]) * (k[i]) * (l[i] ** 2) * s_45 + 2 * (h[i]) * (
                                          k[i] ** 2) * (l[i]) * s_46 + (h[i] ** 2) * (l[i] ** 2) * s_55 + 2 * (
                                          h[i] ** 2) * (k[i]) * (l[i]) * s_56 + (h[i] ** 2) * (k[i] ** 2) * s_66) * (
                                         h[i] ** 2 * k[i] ** 2 + h[i] ** 2 * l[i] ** 2 + k[i] ** 2 * l[i] ** 2) / (
                                         h[i] ** 2 + k[i] ** 2 + l[i] ** 2) ** 2
                    Ehkl_1.append(E_hkl1)
                    Ehkl.append(1 / E_hkl1)
                    # print(1 / E_hkl1)
                    UDEDM_x1 = 5.657 * np.sin(TTH[i] * np.pi / 360) / np.sqrt(1 / E_hkl1) / wavelengt
                    UDEDM_y1 = (FWHM2[i] * np.pi / 180) * np.cos(TTH[i] * np.pi / 360) / wavelengt

                    UDEDM_x.append(UDEDM_x1)
                    UDEDM_y.append(UDEDM_y1)

            valid_indices = [i for i in range(len(UDEDM_x)) if not (np.isnan(UDEDM_x[i]) or np.isnan(UDEDM_y[i]))]

            # Filtra os valores válidos de UDEDM_x e UDEDM_Y
            UDEDM_x_filtered = [UDEDM_x[i] for i in valid_indices]
            UDEDM_y_filtered = [UDEDM_y[i] for i in valid_indices]

            # Realiza o ajuste linear com os valores filtrados
            p5 = np.polyfit(UDEDM_x_filtered, UDEDM_y_filtered, 1)
            # print(p5)

            Dm = formfactor / p5[1]  # Tamanho

            # e_UDEDM = []

            for i in Ehkl:
                e1 = np.sqrt(2 / i) * p5[0]
                e_UDEDM.append(e1)

            e_media = sum(e_UDEDM) / len(e_UDEDM)

            yr = np.polyval(p5, UDEDM_x_filtered)  # Ajuste Linear
            for i in yr:
                yr5.append(i)

            R5 = stats.linregress(UDEDM_x_filtered, UDEDM_y_filtered)  # Parâmetros de confiança
            R5_2 = R5.rvalue ** 2  # Parâmetros de confiança

            # Dados e ajuste
            trace1 = go.Scatter(x=UDEDM_x_filtered, y=UDEDM_y_filtered, mode='markers', name='WHP - UDEDM ')
            trace2 = go.Scatter(x=UDEDM_x_filtered, y=yr5, mode='lines', name='Ajuste', line=dict(color='red'))

            # Título e labels dos eixos
            title_text = r'$<D> = {:.2f} nm     \ \ \ \   \varepsilon = {:.2e}      \ \ \ \        R^2 = {:.4f}$'.format(
                Dm, e_media, R5_2)
            fig = go.Figure(data=[trace1, trace2])
            fig.update_layout(title=title_text, xaxis_title=r'$ 4 \sqrt{2} \sin (\theta) / \lambda \sqrt{E_{hkl}} $',
                              yaxis_title=r'$ \beta \cos (\theta) / \lambda $')
            graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            return render_template('plot.html', graphJSON=graphJSON)

        if modelo_selecionado == "UDSM - SSP":
            for i in range(len(TTH)):

                s_11 = S[0]
                s_12 = S[1]
                s_13 = S[2]
                s_14 = S[3]
                s_15 = S[4]
                s_16 = S[5]

                s_21 = S[6]
                s_22 = S[7]
                s_23 = S[8]
                s_24 = S[9]
                s_25 = S[10]
                s_26 = S[11]

                s_31 = S[12]
                s_32 = S[13]
                s_33 = S[14]
                s_34 = S[15]
                s_35 = S[16]
                s_36 = S[17]

                s_41 = S[18]
                s_42 = S[19]
                s_43 = S[20]
                s_44 = S[21]
                s_45 = S[22]
                s_46 = S[23]

                s_51 = S[24]
                s_52 = S[25]
                s_53 = S[26]
                s_54 = S[27]
                s_55 = S[28]
                s_56 = S[29]

                s_61 = S[30]
                s_62 = S[31]
                s_63 = S[32]
                s_64 = S[33]
                s_65 = S[34]
                s_66 = S[35]

                # Cubic System - All Classes
                if s_11 != 0 and s_12 != 0 and s_44 != 0:
                    E_hkl1 = s_11 - 2 * (s_11 - s_12 - s_44 / 2) * (
                    (h[i] ** 2 * k[i] ** 2 + h[i] ** 2 * l[i] ** 2 + k[i] ** 2 * l[i] ** 2)) / (
                                         h[i] ** 2 + k[i] ** 2 + l[i] ** 2) ** 2

                    Ehkl_1.append(E_hkl1)
                    Ehkl.append(1 / E_hkl1)
                    # print(1 / E_hkl1)

                    d1 = wavelengt / 2 / np.sin(TTH[i] * np.pi / 360)
                    SSP_UDSM_x1 = d1 ** 2
                    SSP_UDSM_y1 = (d1 * (FWHM2[i] * np.pi / 180) * np.cos(TTH[i] * np.pi / 360) / wavelengt) ** 2

                    SSP_UDSM_x.append(SSP_UDSM_x1)
                    SSP_UDSM_y.append(SSP_UDSM_y1)

                # Orthorhombic System - All Classes
                if s_11 != 0 and s_12 != 0 and s_13 != 0 and s_22 != 0 and s_23 != 0 and s_33 != 0 and s_44 != 0 and s_55 != 0 and s_66 != 0:
                    E_hkl1 = ((h[i] ** 4) * s_11 + 2 * (h[i] ** 2) * (k[i] ** 2) * s_12 + 2 * (h[i] ** 2) * (
                                l[i] ** 2) * s_13 + (k[i] ** 4) * s_22 + 2 * (k[i] ** 2) * (l[i] ** 2) * s_23 + (
                                          l[i] ** 4) * s_33 + (k[i] ** 2) * (l[i] ** 2) * s_44 + (h[i] ** 2) * (
                                          l[i] ** 2) * s_55 + (h[i] ** 2) * (l[i] ** 2) * s_66) * (
                             (h[i] ** 2 * k[i] ** 2 + h[i] ** 2 * l[i] ** 2 + k[i] ** 2 * l[i] ** 2)) / (
                                         h[i] ** 2 + k[i] ** 2 + l[i] ** 2) ** 2

                    Ehkl_1.append(E_hkl1)
                    Ehkl.append(1 / E_hkl1)
                    # print(1 / E_hkl1)
                    UDSM_x1 = 4 * np.sin(TTH[i] * np.pi / 360) / Ehkl[i] / wavelengt
                    UDSM_y1 = (FWHM2[i] * np.pi / 180) * np.cos(TTH[i] * np.pi / 360) / wavelengt

                    UDSM_x.append(UDSM_x1)
                    UDSM_y.append(UDSM_y1)

                # Tetragonal System - Classes 4, -4, 4/m
                if s_11 != 0 and s_12 != 0 and s_13 != 0 and s_33 != 0 and s_44 != 0 and s_16 != 0 and s_66 != 0:
                    E_hkl1 = ((h[i] ** 4 + k[i] ** 4) * s_11 + (l[i] ** 4) * s_33 + (h[i] ** 2) * (k[i] ** 2) * (
                                2 * s_12 + s_66) + (l[i] ** 2) * (1 - l[i] ** 2) * (2 * s_13 + s_14) + (
                                          2 * h[i] * k[i]) * (h[i] ** 2 - k[i] ** 2) * s_16) * (
                             (h[i] ** 2 * k[i] ** 2 + h[i] ** 2 * l[i] ** 2 + k[i] ** 2 * l[i] ** 2)) / (
                                         h[i] ** 2 + k[i] ** 2 + l[i] ** 2) ** 2

                    Ehkl_1.append(E_hkl1)
                    Ehkl.append(1 / E_hkl1)
                    # print(1 / E_hkl1)
                    UDSM_x1 = 4 * np.sin(TTH[i] * np.pi / 360) / Ehkl[i] / wavelengt
                    UDSM_y1 = (FWHM2[i] * np.pi / 180) * np.cos(TTH[i] * np.pi / 360) / wavelengt

                    UDSM_x.append(UDSM_x1)
                    UDSM_y.append(UDSM_y1)

                # Trigonal System - Classes 3, -3
                if s_11 != 0 and s_13 != 0 and s_33 != 0 and s_44 != 0 and s_14 != 0 and s_25 != 0:
                    E_hkl1 = ((1 - l[i] ** 2) ** 2 * s_11 + (l[i] ** 4) * s_33 + (l[i] ** 2) * (1 - l[i] ** 2) * (
                                2 * s_13 + s_44) + (2 * k[i] * l[i]) * (3 * h[i] ** 2 - k[i] ** 2) * s_14 + (
                                          2 * h[i] * l[i]) * (3 * k[i] ** 2 - h[i] ** 2) * s_25) * (
                             (h[i] ** 2 * k[i] ** 2 + h[i] ** 2 * l[i] ** 2 + k[i] ** 2 * l[i] ** 2)) / (
                                         h[i] ** 2 + k[i] ** 2 + l[i] ** 2) ** 2

                    Ehkl_1.append(E_hkl1)
                    Ehkl.append(1 / E_hkl1)
                    # print(1 / E_hkl1)
                    UDSM_x1 = 4 * np.sin(TTH[i] * np.pi / 360) / Ehkl[i] / wavelengt
                    UDSM_y1 = (FWHM2[i] * np.pi / 180) * np.cos(TTH[i] * np.pi / 360) / wavelengt

                    UDEDM_x.append(UDSM_x1)
                    UDEDM_y.append(UDSM_y1)

                # Hexagonal System - All Classes
                if s_11 != 0 and s_13 != 0 and s_33 != 0 and s_44 != 0:
                    E_hkl1 = ((1 - l[i] ** 2) ** 2 * s_11 + (l[i] ** 4) * s_33 + (l[i] ** 2) * (1 - l[i] ** 2) * (
                                2 * s_13 + s_44)) * (
                                         h[i] ** 2 * k[i] ** 2 + h[i] ** 2 * l[i] ** 2 + k[i] ** 2 * l[i] ** 2) / (
                                         h[i] ** 2 + k[i] ** 2 + l[i] ** 2) ** 2
                    Ehkl_1.append(E_hkl1)
                    Ehkl.append(1 / E_hkl1)
                    UDSM_x1 = 4 * np.sin(TTH[i] * np.pi / 360) / Ehkl[i] / wavelengt
                    UDSM_y1 = (FWHM2[i] * np.pi / 180) * np.cos(TTH[i] * np.pi / 360) / wavelengt
                    UDSM_x.append(UDSM_x1)
                    UDSM_y.append(UDSM_y1)

                # Triclinic System - All Classes
                if s_11 != 0 and s_12 != 0 and s_13 != 0 and s_14 != 0 and s_15 != 0 and s_16 != 0 and s_21 != 0 and s_22 != 0 and s_23 != 0 and s_24 != 0 and s_25 != 0 and s_26 != 0 and s_33 != 0 and s_44 != 0 and s_55 != 0 and s_66 != 0:
                    E_hkl1 = ((h[i] ** 4) * s_11 + 2 * (h[i] ** 2) * (k[i] ** 2) * s_12 + 2 * (h[i] ** 2) * (
                                l[i] ** 2) * s_13 + 2 * (h[i] ** 2) * (k[i] * l[i]) * s_14 + 2 * (h[i] ** 3) * (
                              l[i]) * s_15 + 2 * (h[i] ** 3) * (k[i]) * s_16 + (k[i] ** 4) * s_22 + 2 * (k[i] ** 2) * (
                                          l[i] ** 2) * s_23 + 2 * (k[i] ** 3) * (l[i]) * s_24 + 2 * (h[i]) * (
                                          k[i] ** 2) * (l[i]) * s_25 + 2 * (h[i]) * (k[i] ** 3) * s_26 + (
                                          l[i] ** 4) * s_33 + 2 * (k[i]) * (l[i] ** 3) * s_34 + 2 * (h[i]) * (
                                          l[i] ** 3) * s_35 + 2 * (h[i]) * (l[i] ** 2) * (k[i]) * s_36 + (k[i] ** 2) * (
                                          l[i] ** 2) * s_44 + 2 * (h[i]) * (k[i]) * (l[i] ** 2) * s_45 + 2 * (h[i]) * (
                                          k[i] ** 2) * (l[i]) * s_46 + (h[i] ** 2) * (l[i] ** 2) * s_55 + 2 * (
                                          h[i] ** 2) * (k[i]) * (l[i]) * s_56 + (h[i] ** 2) * (k[i] ** 2) * s_66) * (
                                         h[i] ** 2 * k[i] ** 2 + h[i] ** 2 * l[i] ** 2 + k[i] ** 2 * l[i] ** 2) / (
                                         h[i] ** 2 + k[i] ** 2 + l[i] ** 2) ** 2
                    Ehkl_1.append(E_hkl1)
                    Ehkl.append(1 / E_hkl1)
                    # print(1 / E_hkl1)
                    UDSM_x1 = 4 * np.sin(TTH[i] * np.pi / 360) / Ehkl[i] / wavelengt
                    UDSM_y1 = (FWHM2[i] * np.pi / 180) * np.cos(TTH[i] * np.pi / 360) / wavelengt

                    UDEDM_x.append(UDSM_x1)
                    UDEDM_y.append(UDSM_y1)

            valid_indices = [i for i in range(len(SSP_UDSM_x)) if
                             not (np.isnan(SSP_UDSM_x[i]) or np.isnan(SSP_UDSM_y[i]))]

            # Filtra os valores válidos de UDEDM_x e UDEDM_Y
            SSP_UDSM_x_filtered = [SSP_UDSM_x[i] for i in valid_indices]
            SSP_UDSM_y_filtered = [SSP_UDSM_y[i] for i in valid_indices]

            # Realiza o ajuste linear com os valores filtrados
            p6 = np.polyfit(SSP_UDSM_x_filtered, SSP_UDSM_y_filtered, 1)
            # print(p5)

            Dm = formfactor / np.sqrt(p6[0] / i)  # Tamanho

            # e_UDSM = []

            for i in Ehkl:
                e1 = np.sqrt(p6[1] / i) / 2
                e_SSP_UDSM.append(e1)

            e_media = sum(e_SSP_UDSM) / len(e_SSP_UDSM)

            yr = np.polyval(p6, SSP_UDSM_x_filtered)  # Ajuste Linear
            for i in yr:
                yr6.append(i)
            R6 = stats.linregress(SSP_UDSM_x_filtered, SSP_UDSM_y_filtered)  # Parâmetros de confiança
            R6_2 = R6.rvalue ** 2  # Parâmetros de confiança

            # Dados e ajuste
            trace1 = go.Scatter(x=SSP_UDSM_x_filtered, y=SSP_UDSM_y_filtered, mode='markers', name='SSP - UDSM ')
            trace2 = go.Scatter(x=SSP_UDSM_x_filtered, y=yr6, mode='lines', name='Ajuste', line=dict(color='red'))

            # Criar a figura e adicionar os traços
            title_text = r'$<D> = {:.2f} nm     \ \ \ \   \varepsilon = {:.2e}      \ \ \ \        R^2 = {:.4f}$'.format(
                Dm, e_media, R6_2)
            fig = go.Figure(data=[trace1, trace2])
            fig.update_layout(title=title_text, xaxis_title=r'$ 4 \sin (\theta) / \lambda E_{hkl} $',
                              yaxis_title=r'$ \beta \cos (\theta) / \lambda $')
            graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

            # Título e labels dos eixos
            title_text = r'$<D> = {:.2f} nm     \ \ \ \   \varepsilon = {:.2e}      \ \ \ \        R^2 = {:.4f}$'.format(
                Dm, e_media, R6_2)
            fig = go.Figure(data=[trace1, trace2])
            fig.update_layout(title=title_text, xaxis_title=r'$ \beta * \cos  / d*^2$',
                              yaxis_title=r'$ (\beta * / d*)^2$')
            graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            return render_template('plot.html', graphJSON=graphJSON)


if __name__ == '__main__':
    app.run(debug=True)
