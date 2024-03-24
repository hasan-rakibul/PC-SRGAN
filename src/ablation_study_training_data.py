import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'nature', 'bright'])

def plot_vline(axs, anno_x, hline_y_all):
    for i, ax in enumerate(axs):
        y_ratio = (hline_y_all[i] - ax.get_ylim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
        for j in range(i, len(anno_x)):
            if j == i:
                ax.axvline(anno_x[i], ymax=y_ratio, color='k', linestyle='--')
                ax.scatter(anno_x[i], hline_y_all[i], marker='o', color='k')
            for ax_ in axs[j+1:]:
                ax_.axvline(anno_x[i], color='k', linestyle='--')

def generate_plot(phy, no_phy):
    
    # for vertical lines
    anno_x = [13, 16.1, 17.7, 17.35]
    hline_y_all = []

    fig, axs = plt.subplots(nrows=len(phy.columns)-1, ncols=1, figsize=(8, 6), sharex=True, gridspec_kw={'hspace': 0})

    for i, column in enumerate(phy.columns):
        if i==len(phy.columns)-1:  # Skip last column
            break
        axs[i].plot(phy[column], label='PC-SRGAN')
        axs[i].plot(no_phy[column], label='SRGAN')
        axs[i].set_ylabel(column)

        axs[i].tick_params(which='minor', length=0) # Remove minor ticks
        
        if i == 0:
            axs[i].legend(loc=(0.85, 0.22))

        if i in [0, 1]:
            hline_y = max(no_phy[column])
        elif i in [2, 3]:
            hline_y = min(no_phy[column])
        
        axs[i].plot([2, 100], [hline_y, hline_y], 'k--')
        hline_y_all.append(hline_y)

        y_range = axs[i].get_ylim()[1] - axs[i].get_ylim()[0]
        yticks_step = y_range/5
        axs[i].set_yticks(np.arange(0, axs[i].get_ylim()[1]+yticks_step, yticks_step))

    # plotting vertical lines
    plot_vline(axs, anno_x, hline_y_all)

    axs[-1].text(min(anno_x)-2, 40, f'{min(anno_x)}\%', color='k', rotation=90)
    axs[-1].text(max(anno_x)+0.5, 40, f'{max(anno_x)}\%', color='k', rotation=90)
    # axs[-1].annotate(f'{min(anno_x)}\%', xy=(0.1655, 0.075), xytext=(0.15, 0.04), xycoords='figure fraction', textcoords='figure fraction',
    #                   arrowprops=dict(facecolor='black', arrowstyle='->'))

    axs[-1].set_xticks(phy.index, phy.index.astype(str))  # Set xticks to index values
    axs[-1].set_xlabel('\% of training data')

    plt.savefig('./results/ablation_study_training_data.pdf', format="pdf", dpi=600, bbox_inches='tight')


def main():
    phy = pd.DataFrame(columns=['PSNR', 'SSIM', 'MSE', 'H1', 'Train+Val time (minutes)'])
    no_phy = pd.DataFrame(columns=['PSNR', 'SSIM', 'MSE', 'H1', 'Train+Val time (minutes)'])
    
    phy.loc[2] = 27.698170159312514,0.7808334120491675,0.03491706332164134,50.216907022906135, 204.08
    no_phy.loc[2] = 24.528395977520248,0.6713362337724,0.043233287846358916,57.25518374599654, 212.24

    phy.loc[5] = 30.163410442500137,0.8291481499478761,0.023871529525555383,34.31176010501838, 226.88
    no_phy.loc[5] = 26.997603906377652,0.7632220420782231,0.02983763961395621,39.14613885640029, 221.88

    phy.loc[10] = 31.813709381397285,0.8608085285719435,0.021897666516239235,31.99104535136833, 249.33
    no_phy.loc[10] = 28.723993267062674,0.8133798946397421,0.023052120184792087,32.372204824094375, 245.44

    phy.loc[20] = 34.78677507704569,0.9430665585816198,0.006761202215650202,11.326713000229017, 315.12
    no_phy.loc[20] = 30.702257757905763,0.8710920072089291,0.01440335867439503,22.355921638989816, 324.21

    phy.loc[40] = 35.46569352812021,0.9500868403147562,0.006677186999730684,10.91664676410475, 449.72
    no_phy.loc[40] = 31.98924285501604,0.9052844110417961,0.011050501899152292,17.802142635958987, 415.55

    phy.loc[60] = 36.54035467319831,0.9584413319807437,0.004667208390626363,8.205895058357372, 540.99
    no_phy.loc[60] = 32.70724028098483,0.9119350467357076,0.010239368591229415,17.074519578558924, 526.46

    phy.loc[80] = 36.4882803148657,0.958638889306943,0.004543519582357408,8.238427216430317, 782.86
    no_phy.loc[80] = 32.65510422555772,0.9084563909742593,0.011240394148237425,18.836875988313714, 777.71

    phy.loc[100] = 37.27479402558142,0.9651788410781005,0.00405540733345567,7.237192213714728, 853.64
    no_phy.loc[100] = 32.44001455102469,0.9083833506067559,0.01110022178424288,18.14477053597181, 810.40

    generate_plot(phy, no_phy)

if __name__ == "__main__":
    main()