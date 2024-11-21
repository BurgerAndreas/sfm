import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
import os


plotfolder = pathlib.Path(__file__).parent.absolute()
plotfolder = os.path.join(plotfolder, "plots")

myrc = {
    "figure.figsize": (8, 6),  # Adjust the figure size as needed
    "font.size": 20,  # Increase font size
    "lines.linewidth": 3.5,  # Thicker lines
    "lines.markersize": 8,  # Thicker lines
    "legend.fontsize": 15,  # Legend font size
    "legend.frameon": False,  # Display legend frame
    "legend.loc": "upper right",  # Adjust legend position
    "axes.spines.right": False,
    "axes.spines.top": False,  # top and right border
    "text.usetex": True,
}

pkmn_type_colors = [
    "#6890F0",  # Water
    "#F08030",  # Fire
    "#78C850",  # Grass
    "#A8B820",  # Bug
    "#A8A878",  # Normal
    "#A040A0",  # Poison
    "#F8D030",  # Electric
    "#E0C068",  # Ground
    "#EE99AC",  # Fairy
    "#C03028",  # Fighting
    "#F85888",  # Psychic
    "#B8A038",  # Rock
    "#705898",  # Ghost
    "#98D8D8",  # Ice
    "#7038F8",  # Dragon
]

# dark, muterd, deep
PALETTE = "deep"

# _cscheme = {
#     "prior": "black",   
#     "flow": "olive",
#     "final": "blue",
# }
# # prettier pastel colors
# _cscheme = {
#     "prior": "#808080",
#     "flow": "#808000",
#     "final": "#008080",
# }
_cscheme = {
    "prior": "#ffb3ba",
    "flow": "#bae1ff",
    "final": "#008080",
}
#b8dbd3 #f7e7b4 #68c4af #96ead7 #f2f6c3

cdict = {
    # https://www.colorhexa.com/ffb347
    # https://www.colorhexa.com/ffb347
    "DEQ1": "#F8A874",  # F8A874 #FBCEB1
    "DEQ2": "#F58238",
    "DEQ": "#F58238",  # F8A874 #FBCEB1
    "DEQ 1,2": "#F58238",  # F8A874 #FBCEB1
    "Fixed-point reuse": "#F58238",  # F8A874 #FBCEB1
    # https://www.picmonkey.com/colors/blue/pastel-blue
    # https://www.picmonkey.com/colors/gray/slate
    "E1": "#AEC6CF",
    "E4": "#5E8D9F",
    "E8": "#476A77",  # 2ca25f 2b8cbe
    "E 1-8": "#476A77",  # 2ca25f 2b8cbe
    "E 1,4,8": "#476A77",  # 2ca25f 2b8cbe
    "E": "#476A77",  # 2ca25f 2b8cbe
    "No fixed-point reuse": "#476A77",  # 2ca25f 2b8cbe
}


######################################################################################
def set_seaborn_style(
    style="whitegrid", palette=PALETTE, context="poster", figsize=(8, 6), font_scale=0.8
):
    sns.set_style(style=style)  # whitegrid white
    sns.set_palette(palette)
    myrc["figure.figsize"] = figsize
    sns.set_context(
        context,  # {paper, notebook, talk, poster}
        font_scale=font_scale,
        rc=myrc,
    )
    
    
def set_style_after(ax, fs=15, legend=True, loc="best", bbox_to_anchor=None):
    # loc: {'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'}
    plt.grid(False)
    plt.grid(which="major", axis="y", linestyle="-", linewidth="1.0", color="lightgray")

    # removes axes spines top and right
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    if legend is True:
        # increase legend fontsize
        ax.legend(fontsize=fs, loc=loc)
        
        if bbox_to_anchor is not None:
            ax.legend(loc="upper right", bbox_to_anchor=bbox_to_anchor, fontsize=fs)

        # remove legend border
        # ax.legend(frameon=False)
        ax.get_legend().get_frame().set_linewidth(0.0)
        # plt.legend().get_frame().set_linewidth(0.0)
    elif legend is None:
        pass
    else:
        try:
            ax.get_legend().remove()
        except:
            pass

    # plt.tight_layout(pad=0.1)
    plt.tight_layout(pad=0)

    return