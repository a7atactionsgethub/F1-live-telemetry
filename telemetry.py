"""
╔══════════════════════════════════════════════════════════════╗
║           F1 FULL DASHBOARD  —  powered by FastF1           ║
╚══════════════════════════════════════════════════════════════╝

Install requirements:
    pip install fastf1 matplotlib pandas numpy scipy

Run:
    python f1_dashboard.py

Controls (top bar):
    Year | Event (e.g. Monza) | Session (R/Q/FP1/FP2/FP3/S)
    Driver1 | Driver2 | [LOAD]  [CLEAR CACHE]  [HELP]
"""

import warnings
warnings.filterwarnings('ignore')

import fastf1
import fastf1.plotting
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button, TextBox
from matplotlib.collections import LineCollection
import numpy as np
import os

# ── Cache ────────────────────────────────────────────────────
os.makedirs('f1_cache', exist_ok=True)
fastf1.Cache.enable_cache('f1_cache')

# ── Theme ────────────────────────────────────────────────────
BG      = '#08080f'
PANEL   = '#0f0f1a'
BORDER  = '#1c1c2e'
ACCENT  = '#e10600'
BLUE    = '#00a8ff'
GREEN   = '#39b54a'
YELLOW  = '#ffd700'
TEXT    = '#f0f0f0'
SUBTEXT = '#6b6b8a'

COMPOUND_COLORS = {
    'SOFT':         '#e10600',
    'MEDIUM':       '#ffd700',
    'HARD':         '#dddddd',
    'INTERMEDIATE': '#39b54a',
    'WET':          '#0067ff',
    'UNKNOWN':      '#555566',
}

matplotlib.rcParams.update({
    'text.color':      TEXT,
    'axes.labelcolor': SUBTEXT,
    'xtick.color':     SUBTEXT,
    'ytick.color':     SUBTEXT,
    'font.family':     'monospace',
})

# ── App state ────────────────────────────────────────────────
state = {
    'session': None,
    'year': 2024, 'event': 'Monza', 'ses': 'R',
    'drv1': 'VER', 'drv2': 'LEC',
}

# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def style(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    ax.tick_params(colors=SUBTEXT, labelsize=6.5)
    ax.grid(color=BORDER, linewidth=0.5, alpha=0.8)
    if title:
        ax.set_title(title, color=TEXT, fontsize=7.5,
                     fontweight='bold', pad=3, fontfamily='monospace')
    if xlabel: ax.set_xlabel(xlabel, color=SUBTEXT, fontsize=6.5)
    if ylabel: ax.set_ylabel(ylabel, color=SUBTEXT, fontsize=6.5)


def err(ax, msg):
    ax.cla()
    ax.set_facecolor(PANEL)
    ax.axis('off')
    ax.text(0.5, 0.5, f'⚠  {msg}', transform=ax.transAxes,
            color=SUBTEXT, ha='center', va='center',
            fontsize=7, fontfamily='monospace', wrap=True)


def drv_color(ses, code, fallback):
    try:
        return fastf1.plotting.get_driver_color(code, ses)
    except Exception:
        return fallback


# ═══════════════════════════════════════════════════════════════
# PANEL RENDERERS
# ═══════════════════════════════════════════════════════════════

def draw_telemetry(ses):
    for ax in TEL_AXES:
        ax.cla()
    try:
        d   = state['drv1']
        lap = ses.laps.pick_driver(d).pick_fastest()
        tel = lap.get_car_data().add_distance()
        lt  = str(lap['LapTime']).split('days ')[-1]
        dist = tel['Distance']

        ax_spd.plot(dist,  tel['Speed'],              color=ACCENT,  lw=1.2)
        ax_thr.plot(dist,  tel['Throttle'],            color=GREEN,   lw=1)
        ax_brk.fill_between(dist, tel['Brake'].astype(float), color=YELLOW, alpha=0.85)
        ax_gear.plot(dist, tel['nGear'],               color=BLUE,    lw=1)
        ax_drs.fill_between(dist, tel['DRS'].astype(float),   color='#bb00ff', alpha=0.85)

        style(ax_spd,  f'{d}  FASTEST LAP  —  {lt}', ylabel='Speed')
        style(ax_thr,  ylabel='Throttle %')
        style(ax_brk,  ylabel='Brake')
        style(ax_gear, ylabel='Gear')
        style(ax_drs,  xlabel='Distance (m)', ylabel='DRS')
        for ax in TEL_AXES[:-1]:
            plt.setp(ax.get_xticklabels(), visible=False)
    except Exception as e:
        err(ax_spd, str(e))


def draw_comparison(ses):
    ax_cmp.cla()
    try:
        d1, d2 = state['drv1'], state['drv2']
        l1 = ses.laps.pick_driver(d1).pick_fastest()
        l2 = ses.laps.pick_driver(d2).pick_fastest()
        t1 = l1.get_car_data().add_distance()
        t2 = l2.get_car_data().add_distance()
        c1 = drv_color(ses, d1, ACCENT)
        c2 = drv_color(ses, d2, BLUE)

        ax_cmp.plot(t1['Distance'], t1['Speed'], color=c1, lw=1.2, label=d1)
        ax_cmp.plot(t2['Distance'], t2['Speed'], color=c2, lw=1.2, label=d2)

        try:
            from scipy.interpolate import interp1d
            f1i = interp1d(t1['Distance'], t1['Speed'], bounds_error=False, fill_value='extrapolate')
            f2i = interp1d(t2['Distance'], t2['Speed'], bounds_error=False, fill_value='extrapolate')
            xc  = np.linspace(0, min(t1['Distance'].max(), t2['Distance'].max()), 500)
            s1c = f1i(xc); s2c = f2i(xc)
            ax_cmp.fill_between(xc, s1c, s2c, where=s1c > s2c, alpha=0.15, color=c1)
            ax_cmp.fill_between(xc, s1c, s2c, where=s2c > s1c, alpha=0.15, color=c2)
        except Exception:
            pass

        style(ax_cmp, f'{d1}  vs  {d2}  —  Speed (km/h)',
              xlabel='Distance (m)', ylabel='Speed')
        ax_cmp.legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT,
                      framealpha=0.8, loc='lower right')
    except Exception as e:
        err(ax_cmp, str(e))


def draw_track(ses):
    ax_track.cla()
    ax_track.set_facecolor(PANEL)
    try:
        lap    = ses.laps.pick_driver(state['drv1']).pick_fastest()
        pos    = lap.get_pos_data()
        tel    = lap.get_car_data().add_distance()
        merged = tel.merge_channels(pos)
        x, y, spd = merged['X'].values, merged['Y'].values, merged['Speed'].values

        pts  = np.array([x, y]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        norm = plt.Normalize(spd.min(), spd.max())
        lc   = LineCollection(segs, cmap='RdYlGn', norm=norm, linewidth=2.5)
        lc.set_array(spd)
        ax_track.add_collection(lc)
        ax_track.set_xlim(x.min()-200, x.max()+200)
        ax_track.set_ylim(y.min()-200, y.max()+200)
        ax_track.set_aspect('equal')
        ax_track.axis('off')
        ax_track.set_title(f'{state["drv1"]}  TRACK SPEED MAP',
                           color=TEXT, fontsize=7.5, fontweight='bold', pad=3)
        cb = fig.colorbar(lc, ax=ax_track, fraction=0.025, pad=0.02)
        cb.ax.tick_params(labelsize=6, colors=SUBTEXT)
        cb.set_label('km/h', color=SUBTEXT, fontsize=6)
    except Exception as e:
        err(ax_track, str(e))


def draw_results(ses):
    ax_res.cla()
    ax_res.set_facecolor(PANEL)
    ax_res.axis('off')
    try:
        want = ['Position','Abbreviation','TeamName','GridPosition','Points','Status','Time']
        cols = [c for c in want if c in ses.results.columns]
        res  = ses.results[cols].head(20)
        ax_res.set_title('RESULTS', color=TEXT, fontsize=7.5, fontweight='bold', pad=3)
        xs = np.linspace(0.02, 0.98, len(cols))

        for j, col in enumerate(cols):
            ax_res.text(xs[j], 0.97, col[:7].upper(),
                        transform=ax_res.transAxes,
                        color=ACCENT, fontsize=5.5, fontweight='bold',
                        ha='center', va='top')

        for i, (_, row) in enumerate(res.iterrows()):
            yp    = 0.91 - i * 0.044
            color = TEXT if i % 2 == 0 else SUBTEXT
            for j, col in enumerate(cols):
                val = str(row[col])
                if col == 'TeamName': val = val[:13]
                elif col == 'Time':   val = str(val).split('days ')[-1][:10]
                else: val = val[:8]
                ax_res.text(xs[j], yp, val,
                            transform=ax_res.transAxes,
                            color=color, fontsize=5.5, ha='center', va='top')
    except Exception as e:
        err(ax_res, str(e))


def draw_laps(ses):
    ax_laps.cla()
    try:
        d1, d2 = state['drv1'], state['drv2']
        c1 = drv_color(ses, d1, ACCENT)
        c2 = drv_color(ses, d2, BLUE)
        for d, c in [(d1, c1), (d2, c2)]:
            lps  = ses.laps.pick_driver(d).pick_quicklaps()
            secs = lps['LapTime'].dt.total_seconds()
            ax_laps.plot(lps['LapNumber'], secs,
                         marker='o', ms=2, lw=1, color=c, label=d)
        style(ax_laps, 'LAP TIMES', xlabel='Lap', ylabel='Time (s)')
        ax_laps.legend(fontsize=6.5, facecolor=PANEL, labelcolor=TEXT, framealpha=0.8)
    except Exception as e:
        err(ax_laps, str(e))


def draw_tyres(ses):
    ax_tyre.cla()
    try:
        laps    = ses.laps
        drivers = ses.drivers[:20]
        abbrevs = []
        for i, drv in enumerate(drivers):
            try:   abbr = ses.get_driver(drv)['Abbreviation']
            except Exception: abbr = str(drv)
            abbrevs.append(abbr)
            drv_laps = laps.pick_driver(drv)
            for _, lap in drv_laps.iterlaps():
                compound = str(lap.get('Compound') or 'UNKNOWN').upper()
                color    = COMPOUND_COLORS.get(compound, '#555566')
                ax_tyre.barh(i, 1, left=lap['LapNumber']-1,
                             color=color, edgecolor='none', height=0.75)
            pit_laps = drv_laps[drv_laps['PitOutTime'].notna()]['LapNumber']
            for pl in pit_laps:
                ax_tyre.axvline(x=pl-1, ymin=(i-0.4)/len(drivers),
                                ymax=(i+0.4)/len(drivers),
                                color='white', lw=0.6, alpha=0.5)

        ax_tyre.set_yticks(range(len(abbrevs)))
        ax_tyre.set_yticklabels(abbrevs, fontsize=6, color=TEXT)
        style(ax_tyre, 'TYRE STRATEGY  (white lines = pit stops)', xlabel='Lap')
        for compound, color in COMPOUND_COLORS.items():
            if compound != 'UNKNOWN':
                ax_tyre.barh(0, 0, color=color, label=compound)
        ax_tyre.legend(fontsize=6, facecolor=PANEL, labelcolor=TEXT,
                       framealpha=0.8, loc='upper right', ncol=5,
                       bbox_to_anchor=(1, 1.18))
    except Exception as e:
        err(ax_tyre, str(e))


def draw_weather(ses):
    ax_wx.cla()
    try:
        w = ses.weather_data
        if w is None or w.empty:
            raise ValueError('No weather data')
        ax_wx.plot(w.index, w['AirTemp'],   color=YELLOW, lw=1,   label='Air °C')
        ax_wx.plot(w.index, w['TrackTemp'], color=ACCENT, lw=1,   label='Track °C')
        ax_wx.plot(w.index, w['Humidity'],  color=BLUE,   lw=0.8, label='Humidity %',
                   alpha=0.7, linestyle='--')
        style(ax_wx, 'WEATHER', ylabel='°C / %')
        ax_wx.legend(fontsize=6, facecolor=PANEL, labelcolor=TEXT, framealpha=0.8)
        ax_rain = ax_wx.twinx()
        ax_rain.fill_between(w.index, w['Rainfall'].astype(float),
                             color=BLUE, alpha=0.25)
        ax_rain.set_ylabel('Rain', color=SUBTEXT, fontsize=6)
        ax_rain.tick_params(colors=SUBTEXT, labelsize=6)
    except Exception as e:
        err(ax_wx, str(e))


def draw_gaps(ses):
    ax_gaps.cla()
    try:
        laps    = ses.laps
        drivers = ses.drivers[:8]
        leader  = drivers[0]
        l_laps  = laps.pick_driver(leader).pick_quicklaps()
        l_cum   = l_laps['LapTime'].dt.total_seconds().cumsum().values
        colors  = plt.cm.tab10(np.linspace(0, 1, len(drivers)))

        for idx, drv in enumerate(drivers[1:]):
            try:
                dl    = laps.pick_driver(drv).pick_quicklaps()
                d_cum = dl['LapTime'].dt.total_seconds().cumsum().values
                n     = min(len(l_cum), len(d_cum))
                gap   = d_cum[:n] - l_cum[:n]
                abbr  = ses.get_driver(drv)['Abbreviation']
                ax_gaps.plot(range(n), gap, lw=1, label=abbr, color=colors[idx+1])
            except Exception:
                pass

        ax_gaps.axhline(0, color=ACCENT, lw=0.8, linestyle='--', alpha=0.6)
        style(ax_gaps, 'GAP TO LEADER', xlabel='Lap', ylabel='Gap (s)')
        ax_gaps.legend(fontsize=6, facecolor=PANEL, labelcolor=TEXT,
                       framealpha=0.8, ncol=4, loc='upper left')
    except Exception as e:
        err(ax_gaps, str(e))


def draw_speed_dist(ses):
    ax_hist.cla()
    try:
        d1, d2 = state['drv1'], state['drv2']
        c1 = drv_color(ses, d1, ACCENT)
        c2 = drv_color(ses, d2, BLUE)
        for d, c in [(d1, c1), (d2, c2)]:
            lap = ses.laps.pick_driver(d).pick_fastest()
            tel = lap.get_car_data()
            ax_hist.hist(tel['Speed'], bins=40, color=c,
                         alpha=0.55, label=d, density=True)
        style(ax_hist, 'SPEED DISTRIBUTION',
              xlabel='Speed (km/h)', ylabel='Density')
        ax_hist.legend(fontsize=6.5, facecolor=PANEL, labelcolor=TEXT, framealpha=0.8)
    except Exception as e:
        err(ax_hist, str(e))


def draw_gear_map(ses):
    ax_gmap.cla()
    ax_gmap.set_facecolor(PANEL)
    try:
        lap    = ses.laps.pick_driver(state['drv1']).pick_fastest()
        pos    = lap.get_pos_data()
        tel    = lap.get_car_data().add_distance()
        merged = tel.merge_channels(pos)
        x, y, gear = merged['X'].values, merged['Y'].values, merged['nGear'].values

        pts  = np.array([x, y]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        norm = plt.Normalize(1, 8)
        lc   = LineCollection(segs, cmap='plasma', norm=norm, linewidth=2.5)
        lc.set_array(gear)
        ax_gmap.add_collection(lc)
        ax_gmap.set_xlim(x.min()-200, x.max()+200)
        ax_gmap.set_ylim(y.min()-200, y.max()+200)
        ax_gmap.set_aspect('equal')
        ax_gmap.axis('off')
        ax_gmap.set_title(f'{state["drv1"]}  GEAR MAP',
                          color=TEXT, fontsize=7.5, fontweight='bold', pad=3)
        cb = fig.colorbar(lc, ax=ax_gmap, fraction=0.025, pad=0.02)
        cb.ax.tick_params(labelsize=6, colors=SUBTEXT)
        cb.set_label('Gear', color=SUBTEXT, fontsize=6)
    except Exception as e:
        err(ax_gmap, str(e))


# ═══════════════════════════════════════════════════════════════
# LOAD
# ═══════════════════════════════════════════════════════════════

def load_session(_=None):
    try:
        state['year']  = int(tb_year.text.strip())
        state['event'] = tb_event.text.strip()
        state['ses']   = tb_ses.text.strip()
        state['drv1']  = tb_d1.text.strip().upper()
        state['drv2']  = tb_d2.text.strip().upper()
    except Exception:
        pass

    set_status(f"Loading  {state['year']}  {state['event']}  {state['ses']} …  (may take a minute on first load)")
    fig.canvas.draw()

    try:
        ses = fastf1.get_session(state['year'], state['event'], state['ses'])
        ses.load()
        state['session'] = ses
        set_status(f"✓  {state['year']}  {state['event']}  {state['ses']}"
                   f"  |  Drivers: {state['drv1']} vs {state['drv2']}"
                   f"  |  {len(ses.laps)} laps loaded")
    except Exception as e:
        set_status(f"Error: {e}")
        fig.canvas.draw()
        return

    draw_telemetry(ses)
    draw_comparison(ses)
    draw_track(ses)
    draw_results(ses)
    draw_laps(ses)
    draw_tyres(ses)
    draw_weather(ses)
    draw_gaps(ses)
    draw_speed_dist(ses)
    draw_gear_map(ses)
    fig.canvas.draw_idle()


def set_status(msg):
    ax_status.cla()
    ax_status.set_facecolor(BORDER)
    ax_status.axis('off')
    ax_status.text(0.01, 0.5, msg, transform=ax_status.transAxes,
                   color=TEXT, fontsize=7.5, va='center', fontfamily='monospace')
    fig.canvas.draw_idle()


def clear_cache(_):
    import shutil
    try:
        shutil.rmtree('f1_cache')
        os.makedirs('f1_cache', exist_ok=True)
        fastf1.Cache.enable_cache('f1_cache')
        set_status('✓  Cache cleared — next load will re-download data')
    except Exception as e:
        set_status(f'Cache error: {e}')


def show_help(_):
    set_status(
        'SESSIONS: R=Race  Q=Qualifying  FP1/FP2/FP3=Practice  S=Sprint  SQ=SprintQual  '
        '|  EVENTS: Monza  Silverstone  Spa  Monaco  Bahrain  Suzuka  Interlagos  '
        '|  DRIVERS: VER LEC HAM NOR SAI RUS ALO PIA STR GAS  (3-letter code)'
    )


# ═══════════════════════════════════════════════════════════════
# BUILD FIGURE
# ═══════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(24, 16), facecolor=BG)
fig.suptitle('F1  FULL  DASHBOARD', color=TEXT,
             fontsize=17, fontweight='bold',
             fontfamily='monospace', y=0.997)

master = gridspec.GridSpec(
    3, 1, figure=fig,
    height_ratios=[0.038, 0.020, 0.942],
    hspace=0.01, top=0.972, bottom=0.01,
    left=0.02, right=0.98
)

# ── Controls ─────────────────────────────────────────────────
ctrl_gs = gridspec.GridSpecFromSubplotSpec(1, 8, subplot_spec=master[0], wspace=0.25)
ax_c = [fig.add_subplot(ctrl_gs[i]) for i in range(8)]

tb_year  = TextBox(ax_c[0], 'Year ',   initial='2024',  color=PANEL, hovercolor=BORDER)
tb_event = TextBox(ax_c[1], 'Event ',  initial='Monza', color=PANEL, hovercolor=BORDER)
tb_ses   = TextBox(ax_c[2], 'Ses ',    initial='R',     color=PANEL, hovercolor=BORDER)
tb_d1    = TextBox(ax_c[3], 'Drv1 ',   initial='VER',   color=PANEL, hovercolor=BORDER)
tb_d2    = TextBox(ax_c[4], 'Drv2 ',   initial='LEC',   color=PANEL, hovercolor=BORDER)
btn_load = Button(ax_c[5], 'LOAD',        color=ACCENT,    hovercolor='#ff3333')
btn_clr  = Button(ax_c[6], 'CLEAR CACHE', color='#1a1a2e', hovercolor=BORDER)
btn_help = Button(ax_c[7], 'HELP',        color='#1a1a2e', hovercolor=BORDER)

for tb in [tb_year, tb_event, tb_ses, tb_d1, tb_d2]:
    tb.label.set_color(SUBTEXT); tb.label.set_fontsize(7)
    tb.text_disp.set_color(TEXT); tb.text_disp.set_fontsize(9)
    tb.text_disp.set_fontfamily('monospace')

btn_load.label.set_color(TEXT); btn_load.label.set_fontweight('bold'); btn_load.label.set_fontsize(9)
btn_clr.label.set_color(SUBTEXT);  btn_clr.label.set_fontsize(7)
btn_help.label.set_color(SUBTEXT); btn_help.label.set_fontsize(8)

# ── Status bar ───────────────────────────────────────────────
ax_status = fig.add_subplot(master[1])
ax_status.set_facecolor(BORDER)
ax_status.axis('off')
ax_status.text(
    0.01, 0.5,
    'Enter  Year / Event / Session / Drivers  then click  LOAD'
    '        |  Sessions: R=Race  Q=Qualifying  FP1/FP2/FP3=Practice  S=Sprint'
    '        |  Click HELP for driver codes',
    transform=ax_status.transAxes,
    color=SUBTEXT, fontsize=7, va='center', fontfamily='monospace'
)

# ── Content grid  4 rows × 4 cols ────────────────────────────
content = gridspec.GridSpecFromSubplotSpec(
    4, 4, subplot_spec=master[2],
    hspace=0.45, wspace=0.30
)

# Row 0: telemetry strip | track map | results
tel_gs = gridspec.GridSpecFromSubplotSpec(5, 1, subplot_spec=content[0, 0:2], hspace=0.06)
ax_spd  = fig.add_subplot(tel_gs[0])
ax_thr  = fig.add_subplot(tel_gs[1], sharex=ax_spd)
ax_brk  = fig.add_subplot(tel_gs[2], sharex=ax_spd)
ax_gear = fig.add_subplot(tel_gs[3], sharex=ax_spd)
ax_drs  = fig.add_subplot(tel_gs[4], sharex=ax_spd)
TEL_AXES = [ax_spd, ax_thr, ax_brk, ax_gear, ax_drs]

ax_track = fig.add_subplot(content[0, 2])
ax_res   = fig.add_subplot(content[0, 3])

# Row 1: comparison | lap times | weather
ax_cmp  = fig.add_subplot(content[1, 0:2])
ax_laps = fig.add_subplot(content[1, 2])
ax_wx   = fig.add_subplot(content[1, 3])

# Row 2: tyre strategy full width
ax_tyre = fig.add_subplot(content[2, 0:4])

# Row 3: gaps | speed distribution | gear map
ax_gaps = fig.add_subplot(content[3, 0:2])
ax_hist = fig.add_subplot(content[3, 2])
ax_gmap = fig.add_subplot(content[3, 3])

# Apply base style & placeholder text
panel_labels = {
    ax_spd:  'SPEED', ax_thr: 'THROTTLE', ax_brk: 'BRAKE',
    ax_gear: 'GEAR',  ax_drs: 'DRS',
    ax_track:'TRACK SPEED MAP',  ax_res:  'RACE RESULTS',
    ax_cmp:  'DRIVER COMPARISON', ax_laps: 'LAP TIMES',
    ax_wx:   'WEATHER',           ax_tyre: 'TYRE STRATEGY',
    ax_gaps: 'GAP TO LEADER',     ax_hist: 'SPEED DISTRIBUTION',
    ax_gmap: 'GEAR MAP',
}
for ax, lbl in panel_labels.items():
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
    ax.tick_params(colors=SUBTEXT, labelsize=6.5)
    ax.text(0.5, 0.5, lbl, transform=ax.transAxes,
            color=BORDER, ha='center', va='center',
            fontsize=9, fontweight='bold', fontfamily='monospace')

ax_track.axis('off')
ax_res.axis('off')
ax_gmap.axis('off')

# ── Wire buttons ─────────────────────────────────────────────
btn_load.on_clicked(load_session)
btn_clr.on_clicked(clear_cache)
btn_help.on_clicked(show_help)

print("F1 Dashboard ready — enter settings and click LOAD")
plt.show()