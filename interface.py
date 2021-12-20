import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import numpy as np
import plotly_express as px
import squarify




#%% Introduction

def entete():
    txt = u"""
    Bienvenue sur notre application permettant de mieux aiguiller les 
    consomateurs sur leurs carences en calcium. L'objectif est donc de pouvoir 
    proposer un large panel de produit bonne qualité nutritionelle globale ainsi 
    qu'un taux de calcium et de vitamine D élevé. Nous proposons également des 
    outils d'analyse nutritionelle pour les profésionnels de la santé, 
    nutritionnistes ou consamateurs curieux.
    Nous utilisons la base de donnée OpenFood Fact qui contient plus de 2 millions 
    de produits. A terme notre objectif et de pouvoir faire un partenariat avec 
    L'organisme de la santé publique
    """
    cols = st.columns(2) # number of columns in each row! = 2
    cols[0].image("openfoodfacts.png", use_column_width=True)
    cols[1].image("spf.png", use_column_width=True) 
    st.title("Santé Calcium Digital")
    st.text(txt)
    
    
    txt2 = u"""
    Le calcium est essentiel pour maintenir des os solides, pour le bon 
    fonctionnement des muscles et du système sanguin, ainsi que pour limiter
    un stockage excessif des lipides dans le tissu adipeux. En cas de manque, 
    les os peuvent se fragiliser jusqu’à développer une ostéoporose, une maladie 
    handicapante et fréquente qui touche près de 39% des femmes de 65 ans en France.
    La cause de la carences en Calcium est multiple : """
    
    txt3 = u""" 
    Carence en vitamine D : 
    L’organisme a besoin d’associer la vitamine D au calcium afin de bien intégrer
    ce dernier. Si vous manquez d’un de ces nutriments, l’autre risque d’être également
    à des niveaux très bas. Si un supplément en calcium vous a été prescrit, vous pouvez
    d’ailleurs demander qu’il soit couplé à un supplément de vitamine D pour une meilleure
    absorption.
    Nous pourons éguiller les consomateurs sur 2 axes :"""
    st.header("La carence en calcium")
    st.text(txt2)
    st.markdown('- Une alimentation peu équilibrée')
    st.markdown("- L'intolérence au lactose")
    st.markdown('- Le régime Vegan')
    st.markdown('- Une carence en vitamine D')
    st.text(txt3)
    st.markdown('- Les produits à haute teneur en Calcium')
    st.markdown('- Les produits à haute teneur en vitamine D')
    st.markdown('- Des inforamtions sur la qualité des produits')
    txt4 = u"""
    Afin de ne pas créer d'autres problèmes en voulant répondre à ces critères 
    nous chercherons à mettre en avant les produits les plus saint possible."""
    st.text(txt4)
    
    
    
def introduction():
    txt_1 = """
A partir du jeu de données d'OpenFood Fact nous avons utiliser plusieurs filtres
sur les variables pertinentes avec notre problèmatique. Nous avons également traiter les 
données pour renplir les valeurs manquantes et abérantes. Cette étape est détailée dans
le but d'être transparant avec les diffents acteurs souhaitant exploiter nos analyses.
"""

    st.title("Contenu de l'application")
    st.header("1- Détail du nettoyage du jeu de données")
    st.text(txt_1)
    
    txt_2 = """
    A partir de notre base de donnée nous fournissons un analyse complète des produits 
    selon leurs compositions et catégories. Cela permettra de tirer des conclusions sur
    les catégories à privilégier pour être en bonne santé tout en faisant le plein de 
    calsium et de vitamine D. 
    Les consomateurs pouront de plus élargir leurs connaissances globales sur la nutrition
    en visualisant les diverses corrélations existantes entre les nutriments.

"""
    st.header("2- Analyse exploratoire des données")
    st.text(txt_2)
    
    txt_3 = """
Nous mettons à disposition un outil pour selctionner les meilleurs produits en selon les 
catégories et le nutriscore. Un descriptif complet du produit sera fourni.

"""
    st.header("3- Selectioner des meilleurs produits")
    st.text(txt_3)
    



#%% Nettoyage


#%% Les analyses
def analyse_univarie(data, cat):
    sns.set_theme(style="whitegrid")
    fig, axs = plt.subplots(ncols=2, figsize = (10,5))
    x = pd.Series(data[cat], name="x variable")
    sns.distplot(x,ax=axs[0]).set(title='Distribution : ' + cat)
    sns.boxplot(x, ax=axs[1]).set(title='Boite à moustache : ' + cat)
    # affichage des indicateurs
    # decalage :
    decalage = 'bien centrée'
    if x.skew() > 0.1:
        decalage = 'Decalage à droite'
    elif x.skew() < -0.1:
        decalage = 'Decalage à gauche'
    # kurtosis
    concentration = 'Concentration nomale'
    if x.kurtosis() > 0.1:
        concentration = 'Distribution concentrée'
    elif x.kurtosis() < -0.1:
        concentration = 'Distribution aplatie'
        
    bord = "-"*40
    bordure = "*"*95
    print(bordure)
    print(bord + cat + bord)
    st.pyplot(fig)
    st.write("moy:\n",x.mean())
    st.write("med:\n",x.median())
    st.write("var:\n",x.var())
    st.write("ect:\n",x.std())
    st.write("skw:\n",x.skew())
    st.write(decalage)
    st.write("kur:\n",x.kurtosis())
    st.write(concentration)

def uni_quali(data, cat):
    data_count_pnns_1 = pd.DataFrame(data[cat].value_counts()).reset_index()
    data_count_pnns_1.loc[data_count_pnns_1[cat] < 15, 'index'] = 'Other countries'
    fig = px.pie(data_count_pnns_1, values=cat, names='index', title=cat)
    st.plotly_chart(fig)
    
def squarefy_(data,catquali,catquanti):
    calsium_mean = data.groupby(catquali).mean()
    moyenne_calcium_par_groupe = calsium_mean[[catquanti]].sort_values(catquanti, ascending=False).reset_index(level=0)
    moyenne_calcium_par_groupe = moyenne_calcium_par_groupe[moyenne_calcium_par_groupe[catquali] != "unknown" ]
    moyenne_calcium_par_groupe = moyenne_calcium_par_groupe.set_index(catquali)
    moyenne_calcium_par_groupe = moyenne_calcium_par_groupe[moyenne_calcium_par_groupe[catquanti] > 0.00001]
    fig = plt.figure(figsize=(20,8))
    titre = "Proportion moyenne de " + catquanti + " selon " + catquali
    fig.suptitle(titre, fontsize=20)
    values = ['{0:0.5f}%'.format(s) for s in moyenne_calcium_par_groupe[catquanti]]
    
    squarify.plot(sizes=moyenne_calcium_par_groupe[catquanti],
                  label=moyenne_calcium_par_groupe.index,
                  value=values,
                  alpha=.8,
                  color=plt.cm.plasma(np.linspace(0.5, 1, len(moyenne_calcium_par_groupe))))

    plt.axis('off')
    st.pyplot(fig)


def quali_quali(data,ppn):
    cont = data[['nutriscore lettre',ppn]].pivot_table(index='nutriscore lettre',columns=ppn,aggfunc=len,margins=True,margins_name="Total").fillna(0)
    plt.figure(figsize=(16,5))
    tx = cont.loc[:,["Total"]]
    ty = cont.loc[["Total"],:]
    n = len(data)
    indep = tx.dot(ty) / n
    
    c = cont.fillna(0) # On remplace les valeurs nulles par 0
    measure = (c-indep)**2/indep
    xi_n = measure.sum().sum()
    table = measure/xi_n
    heat = sns.heatmap(table.iloc[:-1,:-1],annot=c.iloc[:-1,:-1])
    heat.set_title('Correlation ', fontdict={'fontsize':20}, pad=20)
    st.pyplot(heat)
#%% Comparatifs
@st.cache
def quantile_param(serie,n_tranches = 100.0):
    val_quant = np.zeros(int(n_tranches)+1)
    resultat = serie.copy()
    
    for i in range(int(n_tranches)):
        val_quant[i+1] = serie.quantile(q=(float(i+1)/n_tranches))
        
    for pays in serie.index:
        for j in range(int(n_tranches)):
            if val_quant[j]<=serie[pays]<=val_quant[j+1]:
                resultat[pays] = (j+1)
    return resultat


def affichage_produit(data,item):
    image_produit = data[data['product_name'] == item]['image_url'].values[0]
    st.image(image_produit, width = 200)

#%% conclusion


def interface():
    
    data = pd.read_csv('data_finale.csv', sep = ';')
    data = data[~(data['vitamin-d_100g'] == 100)]
    data = data[~(data['calcium_100g'] == 100)]
    var_quanti=['additives_n',\
                'energy-kj_100g',\
                'calcium_100g',\
                'vitamin-d_100g',\
                'proteins_100g',\
                'sugars_100g',\
                'fat_100g',\
                'iron_100g',\
                'fiber_100g',\
                'sodium_100g',\
                'nutriscore_score',\
                'saturated-fat_100g']
    var_quali = ['pnns_groups_1',\
                 'pnns_groups_2',\
                 'nutriscore lettre']
                 
   
    if st.sidebar.checkbox("Présentation", value=True):
        entete()
        introduction()
   
    liste_partie = [' ','Nettoyage', 'Analyse exploratoire', 'Comparateur de produit']
    nom_analyse = st.sidebar.selectbox('Section de la partie',liste_partie)
    
    if nom_analyse == 'Nettoyage':
        st.title('Nettoyage du jeu de données Open Food fact')
        txt_p1_1 = """
        Le jeu de données initial comporte 2026078 produits  et de 187 colonnes 
        d'information."""
        st.text(txt_p1_1)
        lien_url = 'https://world.openfoodfacts.org/'
        off_lien = "[ici]" + '(' + lien_url + ')'
        st.markdown('- Le lien du jeu de données est disponible ' + off_lien)
        lien_def = 'https://world.openfoodfacts.org/data/data-fields.txt'
        def_lien = "[ici]" + '(' + lien_def + ')'
        st.markdown('- Le lien des définitions est diponible ' + def_lien)
        st.text("""
                Ce doccument est très utile car il donne le nom et la définition des colonnes.
                Dans notre cas il y a 187 colonnes donc cela nous évite de faire une exploration
                "à la main" des différentes catégories.  
                Nous avons par exemple :""" )
        st.markdown("- Les pays où l'article est vendu : countries")
        st.markdown("- la catégorie du produit : categories")
        st.markdown("- La valeur énergétique des produits : energy_kj")
        st.markdown("- les aditifs : additives")
        st.markdown("- La composition des produits : Colonnes qui finissent par _100g (112 variables)")
        
        st.header('Démarche du nettoyage')
        st.subheader('Sélection du Pays')
        txt_p1_2 = """
        Le but de notre application et d'aiguiller les résidents Français. La colone 'countries' indique 
        les noms des pays distributeurs des produits. Afin de déterminer la France nous recherchons si 'fr' 
        ou 'Fr' est présent dans countries. En effet plusieurs langues/majuscules sont présentes.
        Finalement nous arrivons avec l'enssemble des nom suivants : """
        liste_fr = ['Frankreich', 'Francia',\
                    'French',\
                    'Fransa',\
                    'francia',\
                    'french' ,\
                    'fransa',\
                    'France' ,\
                    'france' ,\
                    'França',\
                    'Francuska',\
                    'Francja',\
                    'Francija',\
                    'Frankrike' ,\
                    'Franca',\
                    'Francie' ,\
                    'Francija' ,\
                    'Franța' ,\
                    'Frankrig',\
                    'Frakland',\
                    'Francúzsko',\
                    'Frantzia',\
                    'Frantsa' ,\
                    'frankreich',\
                    'Frankrijk']
        st.text(txt_p1_2)
        st.write(liste_fr)
        st.write("""
                 En appliquant un filtre sur ces noms nous arrivons à un jeu de données composé
                 de 613817 produits et de 187 colonnes  """)
        st.subheader('Sélection des variables')
        st.write("""
                 Pour notre étude nous nous focalisons sur le calcium, la vitamine D ainsi que la qualité
                 nutritionnelle des produits. Des informations générales sont également sauvegardés pour 
                 l'application et les analyses""")
                 
        liste_var = ["product_name",\
                    "origins",\
                    "pnns_groups_1",\
                    "pnns_groups_2",\
                    "nutriscore_score",\
                    "categories_tags",\
                    "image_url",\
                    "additives_n",\
                    "energy-kj_100g",\
                    "calcium_100g",\
                    "vitamin-d_100g",\
                    "proteins_100g",\
                    "sugars_100g",\
                    "fat_100g",\
                    "iron_100g",\
                    "fiber_100g",\
                    "sodium_100g",\
                    "saturated-fat_100g"]
        st.write(liste_var)
        st.write('Nous ne conservons ensuite que le produits dont soit le calcium soit la vitamine D est disponible ce qui représente 8924 produits.')
        
        st.subheader('Traiter les valeurs manquantes')
        st.write('''
                 Nous voulons effectuer un remplissage des valeurs manquantes sur les variables quantitatives (Sauf sur le calcium et la vitamine D car nous voulons garder des valeurs exactes)''')
        st.write('Visualisation des valeurs manquantes :')
        st.image("mnso_fr.png", use_column_width=True)
        st.write('Nous utilisons un algorithme KNN pour remplir les valeurs manquantes')
        st.subheader('Traiter les valeurs dupliquées et abérentes')
        st.markdown('- Nous supprimons les valeurs dupliquées : réduction à 8918 produits ')
        st.markdown('- Nous supprimons les valeurs abérantes (<0 et >100) : réduction à 8431 produits ')

        st.subheader('Remplissage du nutriscore')
        st.write("Un dernière étape et le remplissage du nutriscore. Le nutriscore est une note qui permet de quantifier la qualité nutritionnelle d'un produit ce qui est directement en lien avec notre problèmatique.")
        st.write("Nous calculons alors le nutriscore directement à l'aide du tableau suivant, et ajoutons la lettre correspondante dans notre je de données." )
        st.image("nutris.jpg", use_column_width=True) 

        
    if nom_analyse == 'Analyse exploratoire':
        type_analyse = st.sidebar.radio("Type d'analyse",['Univariée', 'Bivariée', 'Multivariée'])
        st.title('Analyse exploratoire du jeu de données nettoyé')
        st.write("""
                 Le but de cette analyse est réveller certaines caractéristiques/corrélations entre les catégories/composants. C'est une étude générale qui montre quels produits sont à privilégier en donnant une vision globale du jeu de données.""")
        if type_analyse == 'Univariée':
            st.header('Analyse univariée')
            st.write("Le but de l'analyse univarié et de donner un apperçu des distribution et des indicateurs statistiques des variables.")
            
            
            st.subheader('Analyse des variables quantitatives')
            var_quanti_loc = st.selectbox('Sélectionner une variable', var_quanti)
            analyse_univarie(data, var_quanti_loc)
            
            st.subheader('Analyse des variables qualitatives')
            var_quali_loc = st.selectbox('Sélectionner une variable', var_quali)
            uni_quali(data,var_quali_loc)
        
        if type_analyse == 'Bivariée':
            st.header('Analyse bivariée')
            st.subheader('Analyse des variables quantitatives/quantitatives')
            st.write('La première étape est de monter la matrice des corrélations. Plus le rapport entre deux variables et élevé plus elles sont corrélées.')
            st.image('heatmap.png')
            st.write('Nous pouvons ensuite selectionner deux variables pour effectuer une régression linéaire ')
            col1, col2 = st.columns(2)
            with col1:
                var_1 = st.selectbox('première variable',var_quanti)
            with col2:
                var_2 = st.selectbox('deuxième variable',var_quanti)
                
            fig = px.scatter(
                data, x= var_1, y= var_2, opacity=0.65,
                trendline='ols', trendline_color_override='darkblue',
                title = 'Droite de régression entre ' + var_1 + " et " +var_2
                )
            st.plotly_chart(fig)
                
            st.subheader('Analyse des variables quantitatives/qualitatives')
            col3, col4 = st.columns(2)
            with col3:
                var_3 = st.selectbox('Variable quantitative',var_quanti)
            with col4:
                var_4 = st.selectbox('Variable qualitative',var_quali)
            
            squarefy_(data,var_4,var_3)
            
            st.write('Il est également important de voir les boites à moustaches selon les catégories ')
            fig = px.box(data[data['calcium_100g'] < 2], x=var_3, y=var_4)
            st.plotly_chart(fig)
            st.subheader('Analyse des variables qualitatives/qualitatives')
            st.write('Ici nous analysons  si le nutriscore (en lettre) varie selon la catégorie de produit')
            st.image('nutrisppn1.png')
            st.image('nutrippn2.png')
            
        if type_analyse == 'Multivariée':
            st.header('Analyse Multivariée')
            st.subheader('Analyse en fonction du nutriscore')
            st.write("Nous proposons à présent d'analyser 2 variables quantitatives selon une autre qualitative")
            col1, col2, col3 = st.columns(3)
            with col1:
                var1 = st.selectbox('Variable quantitative 1 ',var_quanti)
                min_var1 = int(data[var1].min())
                max_var1 = int(data[var1].max())
                masque_var1 = st.slider('Plage de valeurs',min_var1, max_var1,(min_var1,max_var1))
                data_ = data[data[var1] > masque_var1[0]]
                data_ = data[data[var1] < masque_var1[1]]
            with col2:
                var2 = st.selectbox('Variable quantitative 2 ',var_quanti)
                min_var2 = int(data[var2].min())
                max_var2 = int(data[var2].max())
                masque_var2 = st.slider(' Plage de valeurs',min_var2, max_var2, (min_var2, max_var2))
                data_ = data_[data_[var2] > masque_var2[0]]
                data_ = data_[data_[var2] < masque_var2[1]]
            with col3:
                var_3 = st.selectbox('Variable qualitative',var_quali)
            
            fig = px.scatter(data_, x=var1, y=var2, color=var_3, marginal_y="box",
                             marginal_x="box", trendline="ols", template="simple_white")
            st.plotly_chart(fig, use_container_width= True)

            st.subheader('ACP')
            st.write("Nous présentons une analyse en composante principale : Ebloui des valeurs propres, cercles des corrélations, projection des individus")
            st.image('ebloui.png')
            col4, col5 = st.columns(2)
            with col4:
                st.image('f1f2.png')
                st.image('f3f1.png')
                st.image('f3f2.png')
            with col5:
                st.image('pf1f2.png')
                st.image('pf3f1.png')
                st.image('pf3f2.png')
                
            


    if nom_analyse == 'Comparateur de produit':
        st.title('Vous aider à faire le plein de calcium et de vitamine D')
        st.write("Selon la catégorie de votre choix nous vous proposons les 10 meilleurs produits selon le scoring suivant : ")
        st.write("Nous donnons une note entre 0 et 100 en fonction du taux de calcium, à cela nous soustrayons le nutriscore ")
        
        ppnnunique = np.delete(data['pnns_groups_1'].unique(), 0)
        test = st.selectbox('Selectionner une categorie de produit', ppnnunique)
        st.write(type(data['pnns_groups_1'].unique()))
        # Calcul du score
        data_ = data.copy()
        score = quantile_param(data_[data['pnns_groups_1'] == test]['calcium_100g'])
        score = score - data['nutriscore_score']
        data_['score'] = score.values
        data_10 = data_.sort_values(by = ['score'], ascending = False).iloc[:10,:]
        #st.write(data_10)
        nom_prod = st.selectbox('Liste des 10 meilleurs produits pour le calcium', data_10['product_name'].values)
        col11, col10 = st.columns(2)
        with col11:
            affichage_produit(data_10, nom_prod)
        with col10:
            sous_group = data_[data_['product_name'] == nom_prod]['pnns_groups_2'].values[0]
            nutrilettre = data_[data_['product_name'] == nom_prod]['nutriscore lettre'].values[0]
            energie = data_[data_['product_name'] == nom_prod]['energy-kj_100g'].values[0]
            prot = data_[data_['product_name'] == nom_prod]['proteins_100g'].values[0]
            sucre = data_[data_['product_name'] == nom_prod]['sugars_100g'].values[0]
            calcium = data_[data_['product_name'] == nom_prod]['calcium_100g'].values[0]
            st.markdown("- Sous groupe : " + str(sous_group))
            st.markdown("- Nutri Score : " + str(nutrilettre))
            st.markdown("- calcium 100g : " + str(calcium))
            st.markdown("- Energie Kj: " + str(energie))
            st.markdown("- Proteines 100g : " + str(prot))
            st.markdown("- Sucre 100g : " + str(sucre))
            
        scored = quantile_param(data_[data['pnns_groups_1'] == test]['vitamin-d_100g'])
        scored = scored - data['nutriscore_score']
        data_['scored'] = scored.values
        data_10d = data_.sort_values(by = ['scored'], ascending = False).iloc[:10,:]
        #st.write(data_10d)
        nom_prod = st.selectbox('Liste des 10 meilleurs produits pour la vitamine D', data_10d['product_name'].values)
        col11, col10 = st.columns(2)
        with col11:
            affichage_produit(data_10d, nom_prod)
        with col10:
            sous_group = data_[data_['product_name'] == nom_prod]['pnns_groups_2'].values[0]
            nutrilettre = data_[data_['product_name'] == nom_prod]['nutriscore lettre'].values[0]
            energie = data_[data_['product_name'] == nom_prod]['energy-kj_100g'].values[0]
            prot = data_[data_['product_name'] == nom_prod]['proteins_100g'].values[0]
            sucre = data_[data_['product_name'] == nom_prod]['sugars_100g'].values[0]
            calcium = data_[data_['product_name'] == nom_prod]['vitamin-d_100g'].values[0]
            st.markdown("- Sous groupe : " + str(sous_group))
            st.markdown("- Nutri Score : " + str(nutrilettre))
            st.markdown("- vitamine D 100g : " + str(calcium))
            st.markdown("- Energie Kj: " + str(energie))
            st.markdown("- Proteines 100g : " + str(prot))
            st.markdown("- Sucre 100g : " + str(sucre))
            
        
        
    
    
    
    


#%% Lancement 
interface()