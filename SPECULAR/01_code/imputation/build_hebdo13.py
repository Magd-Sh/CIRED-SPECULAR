# -*- coding: utf-8 -*-
"""Point Hebdo 13 — même style que Point_Hebdo_12 (blanc, sobre, Calibri, texte noir)."""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from PIL import Image
import os

BLACK = RGBColor(0x1A, 0x1A, 0x1A)
GREY = RGBColor(0x55, 0x55, 0x55)
FONT = "Calibri"
PLOTS = "/sessions/zealous-eager-dijkstra/mnt/data completion/imputation/outputs/plots"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]

def slide():
    return prs.slides.add_slide(BLANK)

def txt(s, x, y, w, h, runs, size=15, bold=False, color=BLACK, align=PP_ALIGN.LEFT,
        space_after=9, line=1.0):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    if isinstance(runs, str):
        runs = [(runs, {})]
    first = True
    for text, o in runs:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = o.get("align", align)
        p.space_after = Pt(o.get("space_after", space_after))
        p.line_spacing = line
        content = ("•  " + text) if o.get("bullet") else text
        r = p.add_run()
        r.text = content
        f = r.font
        f.name = FONT
        f.size = Pt(o.get("size", size))
        f.bold = o.get("bold", bold)
        f.color.rgb = o.get("color", color)
    return tb

def title(s, t):
    txt(s, 0.7, 0.35, 11.9, 0.9, t, size=29)

def img(s, path, x, y, w=None, h=None):
    iw, ih = Image.open(path).size
    ar = iw / ih
    if w and not h:
        h = w / ar
    elif h and not w:
        w = h * ar
    s.shapes.add_picture(path, Inches(x), Inches(y), Inches(w), Inches(h))
    return w, h

def mini_table(s, x, y, rows, colw, size=12.5):
    t = s.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y),
                           Inches(sum(colw)), Inches(0.42 * len(rows))).table
    for j, w in enumerate(colw):
        t.columns[j].width = Inches(w)
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            c = t.cell(i, j); c.text = cell
            for p in c.text_frame.paragraphs:
                for r in p.runs:
                    r.font.name = FONT; r.font.size = Pt(size); r.font.bold = (i == 0)
                    r.font.color.rgb = BLACK
            c.fill.solid()
            c.fill.fore_color.rgb = RGBColor(0xF2, 0xF2, 0xF2) if i == 0 else RGBColor(0xFF, 0xFF, 0xFF)
    return t

# ---------------------------------------------------------------- 1 · Titre
s = slide()
txt(s, 1.2, 2.6, 10.9, 1.1, "Point Hebdo 13", size=48, align=PP_ALIGN.CENTER)
txt(s, 1.2, 3.8, 10.9, 1.2,
    "Mise au point technique : le « décrochage » des courbes, le nombre de tirages M,\n"
    "et les sources scientifiques derrière ces choix",
    size=19, align=PP_ALIGN.CENTER, color=GREY)

# ------------------------------------------------- 2 · Le décrochage
s = slide()
title(s, "Le « décrochage » des points imputés — ce que c’était")
txt(s, 0.7, 1.35, 12.2, 2.2, [
    ("Sur certaines usines (ex. Ingolstadt 2017–2018, Bruxelles 2023), le point imputé tracé s’écartait légèrement de la ligne d’interpolation. Ce n’était NI une pondération, NI un effet propre à l’usine.", {"space_after": 8}),
    ("Mécanisme = bruit de Monte-Carlo. Le point tracé était la moyenne des M tirages ; chaque tirage interne = interpolation + bruit de moyenne nulle. Avec M fini, la moyenne des bruits n’est pas exactement nulle → un écart de l’ordre de σ / √M (quelques %).", {}),
], size=15)
mini_table(s, 0.7, 3.9,
    [["Ingolstadt 2018", "valeur"],
     ["Interpolation (niveaux) = ligne pointillée", "44 289"],
     ["Interpolation (logs) = base de l’imputation", "44 289"],
     ["Point tracé (moyenne de 10 tirages)", "42 196"]],
    colw=[5.6, 2.0])
txt(s, 8.6, 3.9, 4.1, 2.6, [
    ("À retenir :", {"bold": True, "size": 14.5, "space_after": 5}),
    ("direction aléatoire (haut ou bas) ;", {"bullet": True, "size": 14}),
    ("amplitude ∝ volatilité propre de l’usine ;", {"bullet": True, "size": 14}),
    ("→ 0 quand M augmente. Aucun impact sur la régression.", {"bullet": True, "size": 14}),
], size=14)

# ------------------------------------------------- 3 · La correction (b)
s = slide()
title(s, "Correction (b) : ligne centrale sans bruit")
txt(s, 0.7, 1.3, 5.1, 5.4, [
    ("On trace désormais l’estimation DÉTERMINISTE (fonction point_estimate) : l’interpolation (trous internes) / la marche ancrée (bords), SANS bruit.", {"space_after": 12}),
    ("La bande d’incertitude, elle, continue de venir des tirages — donc l’incertitude affichée est inchangée.", {"space_after": 12}),
    ("Résultat : le point central retombe exactement sur l’interpolation (Ingolstadt 2018 = 44 289), le « décrochage » disparaît, et la bande reste la vraie incertitude d’imputation.", {}),
], size=15)
img(s, os.path.join(PLOTS, "by_plant", "DE_AUDI_INGOLSTADT.png"), 6.1, 1.7, w=6.8)
txt(s, 6.1, 6.05, 6.8, 0.4,
    "Ingolstadt après correction — le point central suit l’interpolation, la bande vient des 50 tirages.",
    size=10.5, color=GREY, align=PP_ALIGN.CENTER)

# ------------------------------------------------- 4 · Le nombre de tirages M
s = slide()
title(s, "Correction (a) : le nombre de tirages M passe de 10 à 50")
txt(s, 0.7, 1.35, 12.2, 2.0, [
    ("Deux usages distincts du nombre de tirages :", {"bold": True, "size": 15, "space_after": 5}),
    ("Estimation ponctuelle (le coefficient) : efficace même à petit M — efficacité (1 + γ/M)⁻¹, γ = fraction d’information manquante (Rubin 1987).", {"bullet": True}),
    ("Écarts-types stables et reproductibles : exigent BEAUCOUP plus de tirages. C’est la vraie raison d’augmenter M.", {"bullet": True}),
], size=14.5)
txt(s, 0.7, 4.2, 6.2, 2.6, [
    ("Règle empirique : M ≈ 100 × FMI", {"bold": True, "size": 15, "space_after": 5}),
    ("(Bodner 2008 ; White, Royston & Wood 2011)", {"size": 13, "color": GREY, "space_after": 8}),
    ("Raffinement — von Hippel 2020 : pour des écarts-types reproductibles, M croît quadratiquement avec le FMI ; procédure en deux temps (un pilote fixe M).", {"size": 14}),
], size=14)
mini_table(s, 7.2, 4.3,
    [["FMI", "M conseillé (Bodner)"],
     ["0,1", "6"], ["0,3", "24"], ["0,5", "59"]],
    colw=[1.7, 3.4], size=13)
txt(s, 7.2, 6.4, 5.6, 0.5,
    "Notre missingness ~55 % → M = 50 est cohérent.", size=12.5, color=GREY)

# ------------------------------------------------- 5 · Données manquantes & FMI
s = slide()
title(s, "Données manquantes exactes — et pourquoi ≠ FMI")
mini_table(s, 0.7, 1.4,
    [["Périmètre (variable emploi)", "manquantes"],
     ["40 usines × 20 ans (fichier complet)", "50,2 %"],
     ["26 usines d’assemblage × 20 ans", "56,2 %"],
     ["26 assemblage, plage active (cellules imputées)", "54,8 %"]],
    colw=[6.4, 2.2])
txt(s, 0.7, 3.9, 12.2, 3.0, [
    ("Attention : « taux de manquants » ≠ FMI. Le FMI (fraction d’information manquante) est propre à CHAQUE estimand (un coefficient donné) et se lit sur la variance inter-/intra-tirages ; il est en général plus FAIBLE que le taux brut de cellules manquantes.", {"bullet": True}),
    ("Autres variables : production ~63 % manquante ; taille véhicule ~20 % manquante (après fallback).", {"bullet": True}),
    ("Fixer M rigoureusement : lancer la régression pilote, lire frac_missing_info renvoyé par rubin.py, puis appliquer Bodner/White (M ≈ 100 × FMI) ou von Hippel (package R howManyImputations). Le FMI d’un coefficient étant < 55 %, M = 50 laisse de la marge.", {"bullet": True}),
], size=14.5)

# ------------------------------------------------- 6 · Les règles de Rubin
s = slide()
title(s, "Les règles de Rubin — origine et formule")
txt(s, 0.7, 1.35, 12.2, 1.6, [
    ("Ce n’est pas un article isolé : la méthode vient du livre de Donald Rubin, « Multiple Imputation for Nonresponse in Surveys » (Wiley, 1987). Une reformulation accessible et citable : Marshall, Altman, Holder & Royston (2009), BMC Medical Research Methodology.", {}),
], size=15)
txt(s, 0.7, 3.1, 12.2, 2.0, [
    ("La formule (celle du code rubin.py) :", {"bold": True, "size": 15, "space_after": 6}),
    ("Estimation combinée = moyenne des M coefficients.", {"bullet": True}),
    ("Variance totale  T = U + (1 + 1/M) · B    —   U = variance INTRA-tirages, B = variance INTER-tirages.", {"bullet": True}),
    ("Écart-type = √T. Le terme B est ce qui fait entrer l’incertitude d’imputation dans les écarts-types.", {"bullet": True}),
], size=14.5)
txt(s, 0.7, 5.9, 12.2, 0.8,
    "Pour un coefficient de régression : règles standard sans transformation (recommandation de Marshall et al. 2009 pour ce cas).",
    size=13, color=GREY)

# ------------------------------------------------- 7 · L'incertitude re-visualisée (50)
s = slide()
title(s, "L’incertitude re-visualisée — désormais 50 tirages")
img(s, os.path.join(PLOTS, "spaghetti_VW.png"), 2.6, 1.3, w=8.1)
txt(s, 0.7, 6.7, 12.2, 0.6,
    "Groupe Volkswagen — avec 50 tirages, l’enveloppe d’incertitude est plus dense et plus stable qu’avec 10 (les écarts-types de Rubin s’en trouvent plus fiables).",
    size=12.5, color=GREY, align=PP_ALIGN.CENTER)

# ------------------------------------------------- 8 · Sources
s = slide()
title(s, "Sources scientifiques")
txt(s, 0.7, 1.4, 12.2, 5.4, [
    ("Rubin, D. B. (1987). Multiple Imputation for Nonresponse in Surveys. Wiley. — origine de la méthode et des règles de combinaison.", {"bullet": True}),
    ("Marshall, Altman, Holder & Royston (2009). BMC Medical Research Methodology, 9:57. — reformulation des règles de Rubin (celles implémentées).", {"bullet": True}),
    ("Bodner, T. E. (2008). Structural Equation Modeling, 15:651–675. — tableau FMI → M (règle M ≈ 100 × FMI).", {"bullet": True}),
    ("White, Royston & Wood (2011). Statistics in Medicine, 30:377–399. — guide pratique MICE ; règle de M.", {"bullet": True}),
    ("Graham, Olchowski & Gilreath (2007). Prevention Science, 8:206–213. — perte de puissance si M trop petit à FMI élevé.", {"bullet": True}),
    ("von Hippel, P. T. (2020). Sociological Methods & Research, 49:699–718. — règle quadratique, procédure en deux temps.", {"bullet": True}),
], size=14)

out = "/sessions/zealous-eager-dijkstra/mnt/data completion/Point_Hebdo_13.pptx"
prs.save(out)
print("saved", out)
