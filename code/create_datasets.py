import ipdb
import time
import pandas as pd
from datetime import timedelta, date
import unidecode
import os
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import itertools
import multiprocessing
import numpy as np

# split a list into evenly sized chunks
def chunks(l, n):
    return [l[i:i+n] for i in range(0, len(l), n)]


pd.set_option("display.max_rows", None, "display.max_columns", None)
import sklearn.metrics as metrics



def daterange(date1, date2):
    for n in range(int ((date2 - date1).days)+1):
        yield date1 + timedelta(n)


def create_department_data():
    df_deps = pd.read_csv('../reg-dep-data/departements-france.csv', delimiter=',')
    print(df_deps.head())
    deps = {}
    for index, row in df_deps.iterrows():
        code_dep = str(row['code_departement'])
        nom_dep = str(row['nom_departement'])
        deps[code_dep] = nom_dep

    df_deps_n = pd.read_csv('../reg-dep-data/france_labels.csv', delimiter=',')
    print(df_deps_n.head())
    deps_n = []
    for index, row in df_deps_n.iterrows():
        nom_dep = str(row['name'])
        deps_n.append(nom_dep)

    new_dtypes = {"dep": object, "jour": object, "P": int, "T":int, "cl_age90": int, "pop": float}
    df_covid_dep = pd.read_csv('../reg-dep-data/sp-pos-quot-dep-2021-10-06-19h08.csv', delimiter=';', dtype=new_dtypes)
    print(df_covid_dep.head())

    # f.write('name,2020-03-10,2020-03-11,2020-03-12,2020-03-13,2020-03-14,2020-03-15,2020-03-16,2020-03-17,2020-03-18,2020-03-19,2020-03-20,2020-03-21,2020-03-22,2020-03-23,2020-03-24,2020-03-25,2020-03-26,2020-03-27,2020-03-28,2020-03-29,2020-03-30,2020-03-31,2020-04-01,2020-04-02,2020-04-03,2020-04-04,2020-04-05,2020-04-06,2020-04-07,2020-04-08,2020-04-09,2020-04-10,2020-04-11,2020-04-12,2020-04-13,2020-04-14,2020-04-15,2020-04-16,2020-04-17,2020-04-18,2020-04-19,2020-04-20,2020-04-21,2020-04-22,2020-04-23,2020-04-24,2020-04-25,2020-04-26,2020-04-27,2020-04-28,2020-04-29,2020-04-30,2020-05-01,2020-05-02,2020-05-03,2020-05-04,2020-05-05,2020-05-06,2020-05-07,2020-05-08,2020-05-09,2020-05-10,2020-05-11,2020-05-12,2020-05-13,2020-05-14,2020-05-15,2020-05-16,2020-05-17,2020-05-18,2020-05-19,2020-05-20,2020-05-21,2020-05-22,2020-05-23,2020-05-24,2020-05-25,2020-05-26\n')

    # start_dt = date(2020, 5, 13)
    # end_dt = date(2021, 10, 3)
    # start_dt = date(2020, 3, 10)
    # end_dt = date(2020, 5, 26)
    # start_dt = date(2020, 9, 1)
    # end_dt = date(2020, 12, 1)
    start_dt = date(2020, 12, 27)
    end_dt = date(2021, 6, 27)

    f = open('../data/france_labels_'+start_dt.strftime("%Y-%m-%d")+'_'+end_dt.strftime("%Y-%m-%d")+'.csv','w')
    f.write('name')
    for dt in daterange(start_dt, end_dt):
        f.write(','+dt.strftime("%Y-%m-%d"))
    f.write('\n')
    dates = []
    covid_data = {}
    for i, dep in enumerate(deps):
        nom_dep = unidecode.unidecode(deps[dep].lower().replace("-","_").replace(" ","_"))
        if nom_dep in deps_n:
            f.write(nom_dep)
            for dt in daterange(start_dt, end_dt):
                # df = dt+d
                # dates.append(dt.strftime("%Y-%m-%d"))
                dat = dt.strftime("%Y-%m-%d")
                covid_data = df_covid_dep[(df_covid_dep['dep']==str(dep))&(df_covid_dep['jour']==dat)&(df_covid_dep['cl_age90']==0)]
                sum_pos = covid_data['P'].item()
                # import ipdb; ipdb.set_trace()
                # for index, row in covid_data.iterrows():
                #     sum_pos += int(row['P'])
                f.write(','+str(sum_pos))
            f.write("\n")

    f.close()


def create_vaccination_department_data():

    start_dt = date(2020, 12, 27)
    end_dt = date(2021, 3, 27)

    # 2013 Rank   Department  Legal population in 1931    Legal population in 1999    Legal population in 2008    Legal population in 2013    Area (km²)  Pop. density (Pop./km²) INSEE Dept. No.
    dep_data = pd.read_csv('../metadata/departements_population.csv', delimiter="\t")

    df_deps = pd.read_csv('../reg-dep-data/departements-france.csv', delimiter=',')
    print(df_deps.head())
    deps = {}
    for index, row in df_deps.iterrows():
        code_dep = str(row['code_departement'])
        nom_dep = str(row['nom_departement'])
        deps[code_dep] = nom_dep

    df_deps_n = pd.read_csv('../pandemic_tgnn/data/France/france_labels.csv', delimiter=',')
    print(df_deps_n.head())
    deps_n = []
    for index, row in df_deps_n.iterrows():
        nom_dep = str(row['name'])
        deps_n.append(nom_dep)

    df_vac_data = pd.read_csv('../metadata/vacsi12-dep-2021-09-30-19h05.csv', delimiter=';')
    print(df_vac_data.head())

    f = open('../metadata/vaccination_1dose_france_deps.csv','w')
    f.write("dep")
    fc = open('../metadata/vaccination_1dose_france_deps_cum.csv','w')
    fc.write("dep")
    for dt in daterange(start_dt, end_dt):
        f.write(','+dt.strftime("%Y-%m-%d"))
        fc.write(','+dt.strftime("%Y-%m-%d"))
    f.write('\n')
    fc.write('\n')
    for i, dep in enumerate(deps):
        nom_dep = unidecode.unidecode(deps[dep].lower().replace("-","_").replace(" ","_"))
        if nom_dep in deps_n:
            pop_row = dep_data[dep_data['Department']==deps[dep]]
            # import ipdb; ipdb.set_trace()
            pop_val = int(pop_row['Legal population in 2013'].item().replace(",",""))

            f.write(nom_dep)
            fc.write(nom_dep)
            for dt in daterange(start_dt, end_dt):
                dat = dt.strftime("%Y-%m-%d")
                covid_data = df_vac_data[(df_vac_data['dep']==str(dep))&(df_vac_data['jour']==dat)]
                for index, row in covid_data.iterrows():
                    vac = int(row['n_dose1'])
                    f.write(","+str(vac))
                    cum_vac = int(row['n_cum_dose1'])
                    # cum_vac = float(row['n_cum_dose1'])/pop_val
                    fc.write(","+str(cum_vac))

            f.write('\n')
            fc.write('\n')
    f.close()
    fc.close()


def extract_sci_data():
    # df_sci = pd.read_csv('sci/gadm1_nuts3_counties_gadm1_nuts3_counties_Aug2020.tsv', delimiter='\t')
    f = open('../sci/sci_france.csv','w')
    f.write('user_loc,fr_loc,scaled_sci\n')
    # for index, row in df_sci.iterrows():
    with open('../sci/gadm1_nuts3_counties_gadm1_nuts3_counties_Aug2020.tsv') as infile:
        for line in infile:
            line = line.split('\t')
            user_loc = str(line[0])
            fr_loc = str(line[1])
            if 'FR' in user_loc and 'FR' in fr_loc:
                scaled_sci = str(line[2])
                f.write(user_loc+","+fr_loc+","+scaled_sci)
    f.close()


def extract_departments_mobility_data():

    # start_dt = date(2020, 9, 1)
    # end_dt = date(2020, 12, 1)

    start_dt = date(2020, 12, 27)
    end_dt = date(2021, 6, 27)

    times = ['0000','0800','1600']

    for dt in daterange(start_dt, end_dt):
        dat = dt.strftime("%Y-%m-%d")
        total_n_crisis = 0
        polygons = {}
        for tim in times:
            # path_name = '../data/13118171318269968_2020-09-01_2020-12-02_csv/13118171318269968_'+dat+'_'+tim+'.csv'
            path_name = '../metadata/13118171318269968_2020-12-27_2021-06-27_csv/13118171318269968_'+dat+'_'+tim+'.csv'
            df_mob = pd.read_csv(path_name, delimiter=',', na_values='')
            df_mob = df_mob.fillna('')
            for index, row in df_mob.iterrows():
                country = str(row["country"])
                if country=='FR':
                    start_polygon = unidecode.unidecode(str(row['start_polygon_name']).lower().replace("-","_").replace(" ","_"))
                    end_polygon = unidecode.unidecode(str(row['end_polygon_name']).lower().replace("-","_").replace(" ","_"))
                    # print(start_polygon)
                    # print(end_polygon)
                    if str(row['n_crisis'])!="":
                        n_crisis = int(row['n_crisis'])
                        tup = (start_polygon, end_polygon)
                        if tup in polygons:
                            polygons[tup] = polygons[tup] + n_crisis
                        else:
                            polygons[tup] = n_crisis

        f = open('../data/graphs_big/vac_FR_'+dat+'.csv','w')
        for polygon in polygons:
            start_polygon = polygon[0]
            end_polygon = polygon[1]
            # tup = (start_polygon, end_polygon)
            f.write(start_polygon+","+end_polygon+","+str(float(polygons[polygon]))+"\n")
        f.close()


def get_france_nuts():
    f = open('../metadata/france_nuts3.csv','w')
    f.write('"Order","Level","Code","Parent","NUTS-Code","Description"\n')
    df_nuts = pd.read_csv('../metadata/NUTS_33_20211007_203154.csv', delimiter=',')
    for index, row in df_nuts.iterrows():
        nuts_code = str(row['NUTS-Code'])
        if 'FR' in nuts_code:
            f.write(str(row["Order"])+","+str(row["Level"])+","+str(row["Code"])+","+str(row["Parent"])+","+str(row["NUTS-Code"])+","+str(row["Description"])+"\n")
    f.close()


def parse_iris_excel():
    df_iris = pd.read_excel('../metadata/reference_IRIS_geo2021.xlsx',sheet_name='Emboitements_IRIS',header=5)
    f = open('../metadata/iris.csv','w', encoding='utf8')
    f.write("CODE_IRIS|LIB_IRIS|TYP_IRIS|GRD_QUART|DEPCOM|LIBCOM|UU2020|REG|DEP\n")
    for index, row in df_iris.iterrows():
        code_iris = str(row['CODE_IRIS'])
        #if code_iris.startswith('75'):
        lib_iris = str(row['LIB_IRIS'])
        typ_iris = str(row['TYP_IRIS'])
        grd = str(row['GRD_QUART'])
        depcom = str(row['DEPCOM'])
        lib = str(row['LIBCOM'])
        uu = str(row['UU2020'])
        reg = str(row['REG'])
        dep = str(row['DEP'])
        f.write(code_iris+'|'+lib_iris+'|'+typ_iris+'|'+grd+'|'+depcom+'|'+lib+'|'+uu+'|'+reg+'|'+dep+'\n')
    f.close()


def create_infra_dep_dataset():

    dtypes = {'iris2019': 'str', 'semaine_glissante':'str', 'clage_65': 'str', 'ti_classe': 'str', 'td_classe': 'str', 'tp_classe':'str'}

    # names = ['sg-iris-opendata-2021-09-06-19h07', 'sg-iris-opendata', 'sg-iris-opendata-202104']
    mypath = '../data/iris_new'
    names = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    names = sorted(names)

    coms = []

    f_sem = open('../processed/iris_dataset_big.csv','w')
    f_sem.write('iris2019,semaine_glissante,clage_65,ti_classe,td_classe,tp_classe\n')

    #iris2019;semaine_glissante;clage_65;ti_classe;td_classe;tp_classe
    #220160000;2020-11-21-2020-11-27;0;[250;500[;[2500;Max];[5;10[
    #290830000;2020-11-21-2020-11-27;0;[0;10[;[1000;1500[;[0;1[

    for name in names:

        if '.csv' in name:
            print(name)

            # data = pd.read_csv("data/sg-iris-opendata-2021-09-06-19h07.csv", dtype=dtypes)
            # data = pd.read_csv(mypath+"/"+name, dtype=dtypes, delimiter=";")
            # col_len = len(data.columns)
            # if col_len==1:
            #     data = pd.read_csv(mypath+"/"+name, dtype=dtypes, delimiter=",")
            #
            # row = data.iloc[0]
            # # import ipdb; ipdb.set_trace()
            # if " " in str(row['iris2019']) or "[" in str(row['iris2019']):
            f = open(mypath+"/"+name, "r")
            lines = f.readlines()
            f.close()

            header = lines[0]
            if "," in header:
                delim = ","
            else:
                delim = ";"

            lines = lines[1:]

            cols = ['iris2019','semaine_glissante','clage_65','ti_classe', 'td_classe', 'tp_classe']

            lst = []
            for line in lines:

                line_split = line.strip().split(delim)
                iris2019 = line_split[0]
                sem = line_split[1]
                clage = line_split[2]
                rest = delim.join(line_split[3:])
                rest_sp = rest.strip("[").strip("\n").split("[")
                rest_split = []
                for res in rest_sp:
                    res = res.strip(delim)
                    if res!="":
                        rest_split.append(res)
                # import ipdb; ipdb.set_trace()
                ti = ""
                if len(rest_split)>0:
                    ti = "["+rest_split[0].replace("[","").replace("]","").replace(";","-")+"]"
                td = ""
                if len(rest_split)>1:
                    td = "["+rest_split[1].replace("[","").replace("]","").replace(";","-")+"]"
                tp = ""
                if len(rest_split)==3:
                    tp = "["+rest_split[2].replace("[","").replace("]","").replace(";","-")+"]"
                lst.append([iris2019, sem, clage, ti, td, tp])

            data = pd.DataFrame(lst, columns=cols)

            # for index, row in df.iterrows():
            # cols = ['iris2019', 'clage_65', 'ti_classe', 'td_classe' ,'tp_classe']
            # for col in cols:
            #     data[col] = data[col].astype('category')

            print(data.describe())
            # import ipdb; ipdb.set_trace()

            # f = open("../processed/"+name+"_stats.txt","w")
            #
            # f.write(str(data.describe())+"\n\n")
            #
            # sel = ['ti_classe', 'td_classe' ,'tp_classe']
            # for col in sel:
            #     f.write(str(data[col].value_counts(normalize=True))+"\n")
            #     f.write(str(data[col].value_counts())+"\n\n")
            #
            # f.write("Count total NaN at each column : \n"+str(data.isnull().sum()))
            # f.close()

            # f = open("../processed/"+name+'_semaine_glissante_new.txt','w')
            # sel = ['semaine_glissante']
            # for col in sel:
            #     f.write(str(data[col].value_counts())+"\n\n")
            # f.close()

            # f = open('paris_dataset.csv','w')
            # for index, row in data.iterrows():
            #     iris = str(row['iris2019'])
            #     if iris.startswith('75'):
            #         f.write(iris+","+row['semaine_glissante']+","+row['ti_classe']+","+row['td_classe']+","+row['tp_classe']+"\n")
            # f.close()

            #found = data.loc[data['iris2019'].str.startswith('75', na=False)]

            # for index, row in data.iterrows():
            for index, row in tqdm(data.iterrows(), total=data.shape[0]):
                iris = str(row['iris2019'])
                semaine = str(row['semaine_glissante'])
                clage = str(row['clage_65'])
                com = iris+"_"+semaine+"_"+clage
                # if com not in coms:
                f_sem.write(iris+","+semaine+","+clage+","+str(row['ti_classe'])+","+str(row['td_classe'])+","+str(row['tp_classe'])+"\n")
                    # coms.append(com)

    f_sem.close()


def final_infra_dataset(paris=False):

    start_dt = date(2020, 12, 21)
    # end_dt = date(2021, 3, 27)
    end_dt = date(2021, 6, 27)

    if paris:
        df_iris = pd.read_csv("../metadata/paris_iris.csv", delimiter=",")
    else:
        df_iris = pd.read_excel('../metadata/reference_IRIS_geo2021.xlsx',sheet_name='Emboitements_IRIS',header=5)

    irises = {}
    for index, row in df_iris.iterrows():
        code_iris = str(row['CODE_IRIS'])
        lib_iris = str(row['LIB_IRIS'])
        typ_iris = str(row['TYP_IRIS'])
        grd = str(row['GRD_QUART'])
        depcom = str(row['DEPCOM'])
        lib = str(row['LIBCOM'])
        uu = str(row['UU2020'])
        reg = str(row['REG'])
        dep = str(row['DEP'])
        irises[code_iris] = lib_iris

    if paris:
        infra_data = pd.read_csv("../processed/paris_dataset.csv", delimiter=",")
    else:
        infra_data = pd.read_csv("../processed/iris_dataset_big.csv", delimiter=",")
    infra_data = infra_data.set_index(['iris2019','semaine_glissante'])

    infra_data = infra_data[~infra_data.index.duplicated(keep='first')]

    # f = open('../data/infra_dep_data_labels_'+start_dt.strftime("%Y-%m-%d")+'_'+end_dt.strftime("%Y-%m-%d")+'.csv','w')
    f = open('../data/infra_dep_data_labels_'+end_dt.strftime("%Y-%m-%d")+'.csv','w')
    f.write('name')
    for dt in daterange(start_dt, end_dt):
        dat_end = dt + timedelta(6)
        if dat_end > end_dt:
            break
        # f.write(','+dt.strftime("%Y-%m-%d")+'-'+dat_end.strftime("%Y-%m-%d"))
        f.write(','+dat_end.strftime("%Y-%m-%d"))
    f.write('\n')
    dates = []
    # covid_data = pd.DataFrame()
    for iris in tqdm(irises):
        iris_clear = unidecode.unidecode(irises[iris].lower().replace("-","_").replace(" ","_"))
        # if iris_clear in irises_n:
        f.write(iris)
        for dt in daterange(start_dt, end_dt):

            # df = dt+d
            # dates.append(dt.strftime("%Y-%m-%d"))
            dat = dt.strftime("%Y-%m-%d")
            dat_end = dt + timedelta(6)
            if dat_end > end_dt:
                break

            # covid_data = infra_data.loc[int(iris)]
            # a = infra_data['iris2019']==int(iris)
            # b = infra_data['semaine_glissante']==(dat+'-'+dat_end.strftime("%Y-%m-%d"))
            try:
                covid_data = infra_data.loc[[(int(iris),dat+'-'+dat_end.strftime("%Y-%m-%d"))]]
                # import ipdb; ipdb.set_trace()
                # covid_data = covid_data.loc[(covid_data['semaine_glissante']==(dat+'-'+dat_end.strftime("%Y-%m-%d")))&(covid_data['clage_65']==0)]
                if covid_data.empty:
                    continue
                else:
                    sum_pos = covid_data['tp_classe'].item()
                    # print(sum_pos)
                    # import ipdb; ipdb.set_trace()
                    # for index, row in covid_data.iterrows():
                    #     sum_pos += int(row['P'])
                    f.write(','+str(sum_pos))
            except:
                pass
        f.write("\n")

    f.close()


def create_dep_dataset():
    import unidecode
    name = 'departements-france.csv'
    deps = pd.read_csv("../reg-dep-data/"+name, delimiter=",")
    deps_dict = {}
    for index, row in deps.iterrows():
        code = str(row['code_departement'])
        new_nom = str(row['nom_departement']).lower().replace("-","_")
        deps_dict[code] = unidecode.unidecode(new_nom)

    f_sem = open('../processed/departments_dataset.csv','w')

    name = 'sp-pos-quot-dep-2021-09-19-19h08.csv'
    # dtypes = {'iris2019': 'str', 'semaine_glissante':'str', 'clage_65': 'str', 'ti_classe': 'str', 'td_classe': 'str', 'tp_classe':'str'}
    data = pd.read_csv("../reg-dep-data/"+name, delimiter=";")
    for index, row in data.iterrows():
        dep = str(row['dep'])
        jour = str(row['jour'])


def extract_infra_paris_dep_mobility_data():

    # start_dt = date(2020, 9, 1)
    # end_dt = date(2020, 12, 1)

    start_dt = date(2020, 12, 27)
    # end_dt = date(2021, 3, 27)
    end_dt = date(2021, 6, 27)

    com_pops = {}
    df_com_pop = pd.read_csv('../metadata/Population_des_communes_Ile-de-France_INSEE.csv')
    for index, row in df_com_pop.iterrows():
        com = int(row['insee'])
        if str(com).startswith('75'):
            pop = int(row['popmun2017'])
            com_pops[com] = pop

    pop_irises = {}
    df_iris_pop = pd.read_excel('../metadata/base-ic-evol-struct-pop-2016.xls',sheet_name='IRIS',header=5)
    for index, row in df_iris_pop.iterrows():
        dep = str(row['DEP'])
        if dep=="75":
            code_iris = str(row['IRIS'])
            pop_iris = int(row['P16_POP'])
            com = int(row['COM'])
            lib_iris = unidecode.unidecode(str(row['LIBIRIS']).lower().replace("-","_").replace(" ","_").replace("_st_","_saint_"))
            pop_irises[lib_iris] = float(pop_iris)/com_pops[com]

    df_iris = pd.read_csv("../metadata/paris_iris.csv", delimiter=",")

    irises = {}
    irises_in = {}
    for index, row in df_iris.iterrows():
        code_iris = str(row['CODE_IRIS'])
        lib_iris = str(row['LIB_IRIS'])
        typ_iris = str(row['TYP_IRIS'])
        grd = str(row['GRD_QUART'])
        depcom = str(row['DEPCOM'])
        libcom = str(row['LIBCOM'])
        uu = str(row['UU2020'])
        reg = str(row['REG'])
        dep = str(row['DEP'])
        # irises[code_iris] = lib_iris
        irises[code_iris] = unidecode.unidecode(lib_iris.lower().replace("-","_").replace(" ","_"))
        # irises_in[code_iris] = libcom
        irises_in[code_iris] = unidecode.unidecode(libcom.replace("Paris"," ").strip().lower().replace("-","_").replace(" ","_"))

    times = ['0000','0800','1600']

    for dt in daterange(start_dt, end_dt):
        dat = dt.strftime("%Y-%m-%d")
        dat_end = dt + timedelta(6)

        if dat_end > end_dt:
            break

        total_n_crisis = 0
        polygons = {}

        for dt_i in daterange(dt, dat_end):
            dt_i = dt_i.strftime("%Y-%m-%d")
            ## regions with 36
            ## tiles with 88
            for tim in times:
                # path_name = '../data/13118171318269968_2020-09-01_2020-12-02_csv/13118171318269968_'+dat+'_'+tim+'.csv'
                path_name = '../data/3680703923136495_2020-12-27_2021-03-28_csv/3680703923136495_'+dt_i+'_'+tim+'.csv'
                df_mob = pd.read_csv(path_name, delimiter=',', na_values='')
                df_mob = df_mob.fillna('')
                for index, row in df_mob.iterrows():
                    # country = str(row["country"])
                    # if country=='FR':
                    start_polygon = unidecode.unidecode(str(row['start_polygon_name']).lower().replace("-","_").replace(" ","_"))
                    end_polygon = unidecode.unidecode(str(row['end_polygon_name']).lower().replace("-","_").replace(" ","_"))
                    # print(start_polygon)
                    # print(end_polygon)
                    if str(row['n_crisis'])!="":
                        n_crisis = int(row['n_crisis'])
                        tup = (start_polygon, end_polygon)
                        if tup in polygons:
                            polygons[tup] = polygons[tup] + n_crisis
                        else:
                            polygons[tup] = n_crisis


        ## now find the number for iris
        iris_mobilities = {}

        pair_irises = list(itertools.product(irises, repeat=2))

        # import ipdb; ipdb.set_trace()
        for pair_iris in pair_irises:
            iris_clear1 = irises[pair_iris[0]]
            # libcom1 = unidecode.unidecode(irises_in[iris1].replace("Paris"," ").strip().lower().replace("-","_").replace(" ","_"))
            libcom1 = irises_in[pair_iris[0]]

            # iris_clear2 = unidecode.unidecode(irises[iris2].lower().replace("-","_").replace(" ","_"))
            iris_clear2 = irises[pair_iris[1]]
            # libcom2 = unidecode.unidecode(irises_in[iris2].replace("Paris"," ").strip().lower().replace("-","_").replace(" ","_"))
            libcom2 = irises_in[pair_iris[1]]
            new_polygon = (iris_clear1, iris_clear2)
            new_polygon2 = (iris_clear2, iris_clear1)
            # if new_polygon not in iris_mobilities:

            for polygon in polygons:
                start_polygon = polygon[0]
                end_polygon = polygon[1]
                start_polygon_clear = polygon[0].replace("paris","").strip("_")
                end_polygon_clear = polygon[1].replace("paris","").strip("_")
                # import ipdb; ipdb.set_trace()

                if libcom1==start_polygon_clear and libcom2==end_polygon_clear:
                # pol = (start_polygon, end_polygon)
                    # if new_polygon2 not in iris_mobilities:
                    iris_mobilities[new_polygon] = polygons[polygon]
                    break

                # if libcom2==start_polygon_clear and libcom1==end_polygon_clear:
                #     # if new_polygon not in iris_mobilities:
                #     iris_mobilities[new_polygon] = polygons[polygon]
                #     break


        f = open('../data/infra_graphs/infra_FR_'+dat+'-'+dat_end.strftime("%Y-%m-%d")+'.csv','w')
        # for polygon in polygons:
        for iris_mobility in iris_mobilities:
            start_polygon = iris_mobility[0]
            end_polygon = iris_mobility[1]

            # pop = pop_irises[start_polygon]
            # pop_dep =
            pop = float(pop_irises[start_polygon])
            # print(start_polygon+" "+end_polygon)
            # print(pop)
            # tup = (start_polygon, end_polygon)
            if pop!=0:
                f.write(start_polygon+","+end_polygon+","+str(float(iris_mobilities[iris_mobility]*pop))+"\n")
        f.close()


def extract_infra_france_dep_mobility_data():

    start_dt = date(2020, 12, 27)
    # end_dt = date(2021, 3, 27)
    end_dt = date(2021, 6, 27)

    dep_pops = {}
    dep_names = {}
    # df_com_pop = pd.read_csv('../metadata/Population_des_communes_Ile-de-France_INSEE.csv')
    df_dep_pop = pd.read_csv('../metadata/departements_population.csv', delimiter="\t")
    for index, row in df_dep_pop.iterrows():
        dep_code = str(row['INSEE Dept. No.'])
        dep_name =  unidecode.unidecode(str(row['Department'])).lower().replace("-","_").replace(" ","_")
        # com_pops[com] = pop
        # pop_row = dep_data[dep_data['Department']==deps[dep]]
        # import ipdb; ipdb.set_trace()
        pop_val = int(str(row['Legal population in 2013']).replace(",",""))
        dep_pops[dep_code] = pop_val
        dep_names[dep_code] = dep_name

    pop_irises = {}
    irises = {}
    irises_in = {}
    df_iris_pop = pd.read_excel('../metadata/base-ic-evol-struct-pop-2016.xls',sheet_name='IRIS',header=5)
    for index, row in df_iris_pop.iterrows():
        dep = str(row['DEP'])

        code_iris = str(row['IRIS'])
        pop_iris = int(row['P16_POP'])
        # com = int(row['COM'])
        dep = str(row['DEP'])
        lib_iris = unidecode.unidecode(str(row['LIBIRIS']).lower().replace("-","_").replace(" ","_").replace("_st_","_saint_"))
        # pop_irises[code_iris] = float(pop_iris)/dep_pops[dep]
        pop_irises[code_iris] = float(pop_iris)
        irises[code_iris] = unidecode.unidecode(lib_iris.lower().replace("-","_").replace(" ","_"))
        irises_in[code_iris] = unidecode.unidecode(dep_names[dep].replace("Paris"," ").strip().lower().replace("-","_").replace(" ","_"))

    # df_iris = pd.read_csv("../metadata/iris.csv", delimiter="|", encoding='utf8')

    # for index, row in df_iris.iterrows():
    #     code_iris = str(row['CODE_IRIS'])
    #     lib_iris = str(row['LIB_IRIS'])
    #     typ_iris = str(row['TYP_IRIS'])
    #     grd = str(row['GRD_QUART'])
    #     depcom = str(row['DEPCOM'])
    #     libcom = str(row['LIBCOM'])
    #     uu = str(row['UU2020'])
    #     reg = str(row['REG'])
    #     dep = str(row['DEP'])
    #     # irises[code_iris] = lib_iris
    #     irises[code_iris] = unidecode.unidecode(lib_iris.lower().replace("-","_").replace(" ","_"))
    #     # irises_in[code_iris] = libcom
    #     # irises_in[code_iris] = unidecode.unidecode(libcom.replace("Paris"," ").strip().lower().replace("-","_").replace(" ","_"))
    #     irises_in[code_iris] = unidecode.unidecode(depcom.replace("Paris"," ").strip().lower().replace("-","_").replace(" ","_"))

    times = ['0000','0800','1600']

    dtr = daterange(start_dt, end_dt)

    # def do_job(job_id, data_slice):
    for dt in dtr:

        dat = dt.strftime("%Y-%m-%d")
        # dat_end = dt + timedelta(6)
        #
        # if dat_end > end_dt:
        #     break

        total_n_crisis = 0
        polygons = {}

        # for dt_i in daterange(dt, dat_end):
        # dt_i = dt_i.strftime("%Y-%m-%d")
        ## regions with 36
        ## tiles with 88
        for tim in times:
            # path_name = '../data/13118171318269968_2020-09-01_2020-12-02_csv/13118171318269968_'+dat+'_'+tim+'.csv'
            # path_name = '../data/3680703923136495_2020-12-27_2021-03-28_csv/3680703923136495_'+dt_i+'_'+tim+'.csv'
            path_name = '../metadata/13118171318269968_2020-12-27_2021-06-27_csv/13118171318269968_'+dat+'_'+tim+'.csv'
            df_mob = pd.read_csv(path_name, delimiter=',', na_values='')
            df_mob = df_mob.fillna('')
            df_mob = df_mob[(df_mob['start_polygon_name']==df_mob['end_polygon_name'])]
            for index, row in df_mob.iterrows():
                # country = str(row["country"])
                # if country=='FR':
                start_polygon = unidecode.unidecode(str(row['start_polygon_name']).lower().replace("-","_").replace(" ","_"))
                end_polygon = unidecode.unidecode(str(row['end_polygon_name']).lower().replace("-","_").replace(" ","_"))
                if start_polygon==end_polygon:
                    # print(start_polygon)
                    # print(end_polygon)
                    if str(row['n_crisis'])!="":
                        n_crisis = int(row['n_crisis'])
                        tup = (start_polygon, end_polygon)
                        if tup in polygons:
                            polygons[tup] = polygons[tup] + n_crisis
                        else:
                            polygons[tup] = n_crisis


        dispatch_jobs(dep_names.keys(), 16, dep_names, df_iris_pop, dep_pops, irises, irises_in, pop_irises, polygons, dat)


def do_job(job_id, data_slice, dep_names, df_iris_pop, dep_pops, irises, irises_in, pop_irises, polygons, dat):

    f = open("../metadata/neigh.txt","r")
    lines = f.readlines()
    f.close()

    neighs_d = {}
    for line in lines:
        line = line.strip()
        spl = line.split("|")
        code_iris_0 = str(spl[0]).strip()
        codes_irises = spl[1].split(",")
        neighs_d[code_iris_0] = codes_irises

    for dep in data_slice:
        dep_name = dep_names[dep]
        dep_irises = df_iris_pop['IRIS'][df_iris_pop['DEP']==dep].values.tolist()

        pair_irises = []
        for dep_iris in dep_irises:
            if dep_iris in neighs_d:
                neighs = neighs_d[str(dep_iris)]
                for neigh in neighs:
                    neigh = neigh.strip()
                    pair_irises.append([str(dep_iris), str(neigh)])
                    pair_irises.append([str(neigh), str(dep_iris)])

        # dep_irises = [unidecode.unidecode(lib_iris.lower().replace("-","_").replace(" ","_")) for lib_iris in dep_irises]
        # import ipdb; ipdb.set_trace()
        ## now find the number for iris
        iris_mobilities = {}

        # pair_irises = list(itertools.product(dep_irises, repeat=2))

        # import ipdb; ipdb.set_trace()
        for pair_iris in pair_irises:
            # iris_clear1 = irises[pair_iris[0]]
            iris_clear1 = pair_iris[0]
            # libcom1 = unidecode.unidecode(irises_in[iris1].replace("Paris"," ").strip().lower().replace("-","_").replace(" ","_"))
            try:
                libcom1 = irises_in[pair_iris[0]]

                # iris_clear2 = unidecode.unidecode(irises[iris2].lower().replace("-","_").replace(" ","_"))
                # iris_clear2 = irises[pair_iris[1]]
                iris_clear2 = pair_iris[1]
                # libcom2 = unidecode.unidecode(irises_in[iris2].replace("Paris"," ").strip().lower().replace("-","_").replace(" ","_"))
                libcom2 = irises_in[pair_iris[1]]
                new_polygon = (iris_clear1, iris_clear2)
            except:
                continue
            # new_polygon2 = (iris_clear2, iris_clear1)
            # if new_polygon not in iris_mobilities:

            for polygon in polygons:
                start_polygon = polygon[0]
                end_polygon = polygon[1]
                start_polygon_clear = polygon[0]
                end_polygon_clear = polygon[1]
                # import ipdb; ipdb.set_trace()

                if libcom1==start_polygon_clear and libcom2==end_polygon_clear:
                # pol = (start_polygon, end_polygon)
                    # if new_polygon2 not in iris_mobilities:
                    iris_mobilities[new_polygon] = polygons[polygon]
                    break

                # if libcom2==start_polygon_clear and libcom1==end_polygon_clear:
                #     # if new_polygon not in iris_mobilities:
                #     iris_mobilities[new_polygon] = polygons[polygon]
                #     break

        # directory = '../data/infra_graphs/'+dep_name
        directory = "C:/Users/skian/Documents/ihu-covid/data/infra_graphs_big/"+dep_name
        if not os.path.exists(directory):
            os.makedirs(directory)
        f = open(directory+'/infra_FR_'+dat+'.csv','w')
        # for polygon in polygons:
        for iris_mobility in iris_mobilities:
            start_polygon = iris_mobility[0]
            end_polygon = iris_mobility[1]

            # pop = pop_irises[start_polygon]
            pop_dep = dep_pops[dep]
            pop = float(pop_irises[start_polygon])/pop_dep
            # print(start_polygon+" "+end_polygon)
            # print(pop)
            # tup = (start_polygon, end_polygon)
            if pop!=0:
                f.write(start_polygon+","+end_polygon+","+str(float(iris_mobilities[iris_mobility]*pop))+"\n")
        f.close()


def dispatch_jobs(data, job_number, dep_names, df_iris_pop, dep_pops, irises, irises_in, pop_irises, polygons, dat):
    total = len(data)
    # chunk_size = total / job_number
    # import ipdb; ipdb.set_trace()
    # slice = chunks(data, chunk_size)
    slice = np.array_split(list(data), job_number)
    jobs = []

    for i, s in enumerate(slice):
        j = multiprocessing.Process(target=do_job, args=(i, s, dep_names, df_iris_pop, dep_pops, irises, irises_in, pop_irises, polygons, dat))
        jobs.append(j)
    for j in jobs:
        j.start()





def parse_epci_excel():
    df_iris = pd.read_excel('../metadata/Liste des groupements - France entière.xlsx', engine='openpyxl')
    f = open('../metadata/epci.csv','w', encoding='utf8')
    f.write("code|lib|dep_code|dep_lib|reg|pop\n")
    codes = []
    for index, row in df_iris.iterrows():
        code = str(row['N° SIREN'])
        #if code_iris.startswith('75'):
        lib = str(row['Nom du groupement'])
        # typ = str(row['TYP_IRIS'])
        # grd = str(row['GRD_QUART'])
        dep = str(row['Département siège']).split("-")
        dep_code = dep[0].strip()
        dep_lib = dep[1].strip()
        # uu = str(row['UU2020']
        pop = str(row['Population'])
        reg = str(row['Région siège'])
        if code not in codes:
            f.write(code+'|'+lib+'|'+dep_code+'|'+dep_lib+'|'+reg+'|'+pop+'\n')
            codes.append(code)
    f.close()


def create_epci_dataset():
    dtypes = {'epci2020': 'str', 'nature': 'str', 'semaine_glissante':'str', 'clage_65': 'str', 'ti_classe': 'str', 'td_classe': 'str', 'tp_classe':'str'}

    # names = ['sg-iris-opendata-2021-09-06-19h07', 'sg-iris-opendata', 'sg-iris-opendata-202104']
    mypath = '../data/epci'
    names = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    names = sorted(names)

    coms = []

    f_sem = open('../processed/epci_dataset.csv','w')
    f_sem.write('epci2020,semaine_glissante,clage_65,ti_classe,td_classe,tp_classe\n')
    epcis = []
    for name in names:

        if '.csv' in name:
            print(name)

            f = open(mypath+"/"+name, "r")
            lines = f.readlines()
            f.close()

            header = lines[0]

            delim = ","

            lines = lines[1:]

            cols = ['epci2020','semaine_glissante','clage_65','ti_classe', 'td_classe', 'tp_classe']

            lst = []
            for line in lines:

                line_split = line.strip().split(delim)
                epci2020 = line_split[0].strip("\n")
                nature = line_split[1]
                sem = line_split[2].strip("\n")
                clage = line_split[3]
                rest = delim.join(line_split[4:])
                rest_sp = rest.strip("[").strip("\n").split("[")
                rest_split = []
                for res in rest_sp:
                    res = res.strip(delim)
                    if res!="":
                        rest_split.append(res)
                # import ipdb; ipdb.set_trace()
                ti = ""
                if len(rest_split)>0:
                    ti = "["+rest_split[0].replace("[","").replace("]","").replace(";","-")+"]"
                td = ""
                if len(rest_split)>1:
                    td = "["+rest_split[1].replace("[","").replace("]","").replace(";","-")+"]"
                tp = ""
                if len(rest_split)==3:
                    tp = "["+rest_split[2].replace("[","").replace("]","").replace(";","-")+"]"
                lst.append([epci2020, sem, clage, ti, td, tp])

            data = pd.DataFrame(lst, columns=cols)

            # for index, row in df.iterrows():
            # cols = ['iris2019', 'clage_65', 'ti_classe', 'td_classe' ,'tp_classe']
            # for col in cols:
            #     data[col] = data[col].astype('category')

            print(data.describe())
            # import ipdb; ipdb.set_trace()

            # for index, row in data.iterrows():
            for index, row in tqdm(data.iterrows(), total=data.shape[0]):
                epci = str(row['epci2020'])
                # nature = str(row['nature'])
                semaine = str(row['semaine_glissante'])
                clage = str(row['clage_65'])
                com = epci+"_"+semaine+"_"+clage
                # if com not in coms:
                f_sem.write(epci+","+semaine+","+clage+","+str(row['ti_classe'])+","+str(row['td_classe'])+","+str(row['tp_classe'])+"\n")
                if epci not in epcis:
                    epcis.append(epci)

    f_sem.close()

    f = open('../metadata/epcis.txt',"w")
    for epci in epcis:
        f.write(epci.replace("\n","")+"\n")
    f.close()


def final_epci_dataset():

    start_dt = date(2020, 12, 21)
    end_dt = date(2021, 3, 27)

    # df_epci = pd.read_excel('../metadata/Liste des groupements - France entière.xlsx', engine='openpyxl')
    df_epci = pd.read_excel('../metadata/Intercommunalite-Metropole_au_01-01-2021.xlsx', engine='openpyxl', header=5)
    # EPCI	LIBEPCI	NATURE_EPCI	NB_COM

    epcis_c = {}
    for index, row in df_epci.iterrows():
        try:
            code_epci = int(str(row['EPCI']).strip())
            # code_epci = str(row['N° SIREN']).strip()
            # lib_epci = str(row['Nom du groupement']).strip()
            #if code_iris.startswith('75'):
            lib_epci = str(row['LIBEPCI']).strip()
            # typ = str(row['TYP_IRIS'])
            # grd = str(row['GRD_QUART'])
            # dep = str(row['Département siège']).split("-")
            # dep_code = dep[0].strip()
            # dep_lib = dep[1].strip()
            # uu = str(row['UU2020'])
            # reg = str(row['Région siège'])

            epcis_c[code_epci] = lib_epci
        except:
            pass

    # f = open('../metadata/epcis.txt','r')
    # lines = f.readlines()
    # f.close()
    # epcis = [line.strip() for line in lines]

    infra_data = pd.read_csv("../processed/epci_dataset.csv", delimiter=",")
    infra_data = infra_data.set_index(['epci2020','semaine_glissante'])

    infra_data = infra_data[~infra_data.index.duplicated(keep='first')]

    # f = open('../data/infra_dep_data_labels_'+start_dt.strftime("%Y-%m-%d")+'_'+end_dt.strftime("%Y-%m-%d")+'.csv','w')
    f = open('../data/epci_data_labels_'+end_dt.strftime("%Y-%m-%d")+'.csv','w')
    f.write('name')
    for dt in daterange(start_dt, end_dt):
        dat_end = dt + timedelta(6)
        if dat_end > end_dt:
            break
        # f.write(','+dt.strftime("%Y-%m-%d")+'-'+dat_end.strftime("%Y-%m-%d"))
        f.write(','+dat_end.strftime("%Y-%m-%d"))
    f.write('\n')
    dates = []
    # covid_data = pd.DataFrame()
    for epci in tqdm(list(epcis_c.keys())):
        # epci_clear = unidecode.unidecode(epcis_c[epci].lower().replace("-","_").replace(" ","_"))
        # if iris_clear in irises_n:
        f.write(str(epci))
        for dt in daterange(start_dt, end_dt):

            # df = dt+d
            # dates.append(dt.strftime("%Y-%m-%d"))
            dat = dt.strftime("%Y-%m-%d")
            dat_end = dt + timedelta(6)
            if dat_end > end_dt + timedelta(5):
                break

            # covid_data = infra_data.loc[int(iris)]
            # a = infra_data['iris2019']==int(iris)
            # b = infra_data['semaine_glissante']==(dat+'-'+dat_end.strftime("%Y-%m-%d"))
            try:
                covid_data = infra_data.loc[[(int(epci),dat+'-'+dat_end.strftime("%Y-%m-%d"))]]
                sum_pos = ""
                # import ipdb; ipdb.set_trace()
                # covid_data = covid_data.loc[(covid_data['semaine_glissante']==(dat+'-'+dat_end.strftime("%Y-%m-%d")))&(covid_data['clage_65']==0)]
                if covid_data.empty:
                    continue
                else:
                    sum_pos = covid_data['tp_classe'].item()
                    # print(sum_pos)
                    # import ipdb; ipdb.set_trace()
                    # for index, row in covid_data.iterrows():
                    #     sum_pos += int(row['P'])
                    f.write(','+str(sum_pos))
            except:
                continue
        f.write("\n")

    f.close()


def extract_epci_mobility_data():

    start_dt = date(2020, 12, 27)
    end_dt = date(2021, 3, 27)

    dep_pops = {}
    dep_names = {}
    # df_com_pop = pd.read_csv('../metadata/Population_des_communes_Ile-de-France_INSEE.csv')
    df_dep_pop = pd.read_csv('../metadata/departements_population.csv', delimiter="\t")
    for index, row in df_dep_pop.iterrows():
        dep_code = str(row['INSEE Dept. No.']).strip()
        dep_name =  unidecode.unidecode(str(row['Department'])).lower().replace("-","_").replace(" ","_").strip().replace("\n","")
        # com_pops[com] = pop
        # pop_row = dep_data[dep_data['Department']==deps[dep]]
        # import ipdb; ipdb.set_trace()
        pop_val = int(str(row['Legal population in 2013']).replace(",",""))
        dep_pops[dep_code] = pop_val
        dep_names[dep_code] = dep_name

    epcis_pops = {}
    epcis = {}
    epcis_in = {}

    df_epci_pops = pd.read_excel('../metadata/Liste des groupements - France entière.xlsx')
    # df_epci_pops = pd.read_csv('../metadata/epci.csv', delimiter='|')
    # f = open('../metadata/epci_liste.csv','w')
    # f.write("epci|dep|pop\n")
    for index, row in df_epci_pops.iterrows():
        # epci_code = str(row['code']).strip()
        # epci_name = unidecode.unidecode(str(row['lib'])).lower().replace("-","_").replace(" ","_").strip()

        epci_code = str(row['N° SIREN']).strip().replace("\n","")
        epci_name = str(row['Nom du groupement']).strip().replace("\n","")

        dep = str(row['Département siège']).split(" - ")
        dep_code = str(dep[0]).strip().replace("\n","")
        # dep_code = str(row['dep_code']).strip()
        #dep_lib =

        # pop_val = int(str(row['pop']).replace(",","").strip())
        pop_val = int(str(row['Population']).replace(",","").strip())
        epcis_pops[epci_code] = pop_val

        # lib_iris = unidecode.unidecode(str(row['LIBIRIS']).lower().replace("-","_").replace(" ","_").replace("_st_","_saint_"))
        # epcis_pops[epci_code] = float(pop_val)/dep_pops[dep_code]
        epcis[epci_code] = epci_name
        epcis_in[epci_code] = unidecode.unidecode(dep_names[dep_code].strip().lower().replace("-","_").replace(" ","_"))
        # if dep_code=="75":
        #     print(epci_code)

    #     f.write(str(epci_code)+"|"+str(dep_code)+"|"+str(pop_val)+"\n")
    # f.close()

    # import ipdb; ipdb.set_trace()

    # df_com_pop = pd.read_csv('../metadata/Population_des_communes_Ile-de-France_INSEE.csv')
    df_epci_pop = pd.read_csv('../metadata/epci_liste.csv', delimiter='|')

    times = ['0000','0800','1600']

    dtr = daterange(start_dt, end_dt)

    # def do_job(job_id, data_slice):
    for dt in dtr:

        dat = dt.strftime("%Y-%m-%d")
        # dat_end = dt + timedelta(6)
        #
        # if dat_end > end_dt:
        #     break

        total_n_crisis = 0
        polygons = {}

        # for dt_i in daterange(dt, dat_end):
        # dt_i = dt_i.strftime("%Y-%m-%d")
        ## regions with 36
        ## tiles with 88
        for tim in times:
            # path_name = '../data/13118171318269968_2020-09-01_2020-12-02_csv/13118171318269968_'+dat+'_'+tim+'.csv'
            # path_name = '../data/3680703923136495_2020-12-27_2021-03-28_csv/3680703923136495_'+dt_i+'_'+tim+'.csv'
            path_name = '../metadata/13118171318269968_2020-12-27_2021-03-28_csv/13118171318269968_'+dat+'_'+tim+'.csv'
            df_mob = pd.read_csv(path_name, delimiter=',', na_values='')
            df_mob = df_mob.fillna('')
            df_mob = df_mob[df_mob['country']=='FR']
            df_mob = df_mob[(df_mob['start_polygon_name']==df_mob['end_polygon_name'])]
            for index, row in df_mob.iterrows():
                # country = str(row["country"])
                # if country=='FR':
                start_polygon = unidecode.unidecode(str(row['start_polygon_name']).lower().replace("-","_").replace(" ","_").replace("\n","")).strip()
                end_polygon = unidecode.unidecode(str(row['end_polygon_name']).lower().replace("-","_").replace(" ","_").replace("\n","")).strip()
                if start_polygon==end_polygon:
                    # print(start_polygon)
                    # print(end_polygon)

                    if str(row['n_crisis'])!="":
                        n_crisis = int(row['n_crisis'])
                        tup = (start_polygon, end_polygon)

                        if tup in polygons:
                            polygons[tup] = polygons[tup] + n_crisis
                        else:
                            polygons[tup] = n_crisis

        #import ipdb; ipdb.set_trace()

        f = open("../metadata/epci_neigh_new.txt","r")
        lines = f.readlines()
        f.close()

        neighs_d = {}
        for line in lines:
            line = line.strip()
            spl = line.split("|")
            code_iris_0 = str(spl[0]).strip().replace("\n","")
            if "," in spl[1]:
                codes_irises = spl[1].split(",")
            else:
                codes_irises = [spl[1]]

            if len(codes_irises)>0:
                codes_irises = [c.strip() for c in codes_irises]
                # print(codes_irises)
                neighs_d[code_iris_0] = codes_irises

        #import ipdb; ipdb.set_trace()

        dispatch_jobs_epci(dep_names.keys(), 16, dep_names, df_epci_pop, dep_pops, epcis_in, epcis_pops, polygons, neighs_d, dat)


def do_job_epci(job_id, data_slice, dep_names, df_epci_pop, dep_pops, epcis_in, epcis_pops, polygons, neighs_d, dat):

    for dep in data_slice:
        dep_name = dep_names[str(dep)]
        dep_epcis = df_epci_pop['epci'][df_epci_pop['dep']==str(dep)].values.tolist()
        # if dep=="16":
        #     print(dep_epcis)
        # print(neighs_d)

        pair_irises = []
        for dep_epci in dep_epcis:
            # dep_epci = dep_epci.strip()
            str_dep_epci = str(dep_epci).strip().replace("\n","")
            if str_dep_epci in neighs_d:
                neighs = neighs_d[str_dep_epci]
                # if dep=="16":
                #     print(str_dep_epci)
                for neigh in neighs:
                    neigh = neigh.strip().replace("\n","")
                    if neigh!="" and str_dep_epci!="":
                        # print(neigh)
                        pair_irises.append((str_dep_epci, str_dep_epci))
                        pair_irises.append((str_dep_epci, neigh))
            else:
                if str_dep_epci!="":
                    pair_irises.append((str_dep_epci, str_dep_epci))
                        #pair_irises.append((neigh, str_dep_epci))
                    # print(str(neigh)+" "+str(dep_epci))
        # dep_irises = [unidecode.unidecode(lib_iris.lower().replace("-","_").replace(" ","_")) for lib_iris in dep_irises]
        # import ipdb; ipdb.set_trace()
        ## now find the number for iris
        iris_mobilities = {}

        # pair_irises = list(itertools.product(dep_irises, repeat=2))

        # import ipdb; ipdb.set_trace()
        for pair_iris in pair_irises:
            # iris_clear1 = irises[pair_iris[0]]
            iris_clear1 = pair_iris[0].replace("\n","")
            # libcom1 = unidecode.unidecode(irises_in[iris1].replace("Paris"," ").strip().lower().replace("-","_").replace(" ","_"))
            try:
                libcom1 = epcis_in[pair_iris[0].strip().replace("\n","")].strip().replace("\n","")

                # iris_clear2 = unidecode.unidecode(irises[iris2].lower().replace("-","_").replace(" ","_"))
                # iris_clear2 = irises[pair_iris[1]]
                iris_clear2 = pair_iris[1].strip().replace("\n","")
                # libcom2 = unidecode.unidecode(irises_in[iris2].replace("Paris"," ").strip().lower().replace("-","_").replace(" ","_"))
                libcom2 = epcis_in[pair_iris[1].strip()].strip().replace("\n","")
                new_polygon = (iris_clear1, iris_clear2)
            except:
                 continue
            # new_polygon2 = (iris_clear2, iris_clear1)
            # if new_polygon not in iris_mobilities:

            for polygon in polygons:
                start_polygon = polygon[0].replace("\n","").strip()
                end_polygon = polygon[1].replace("\n","").strip()
                start_polygon_clear = polygon[0].replace("\n","").strip()
                end_polygon_clear = polygon[1].replace("\n","").strip()
                # import ipdb; ipdb.set_trace()

                # if libcom1=="charente":
                #     print("lib1: "+libcom1)
                #     print(start_polygon_clear)
                #     print("lib2: "+libcom2)
                #     print(end_polygon_clear)

                if libcom1==start_polygon_clear and libcom2==end_polygon_clear:
                # pol = (start_polygon, end_polygon)
                    # if new_polygon2 not in iris_mobilities:
                    iris_mobilities[new_polygon] = polygons[polygon]
                    break

                # if libcom2==start_polygon_clear and libcom1==end_polygon_clear:
                #     # if new_polygon not in iris_mobilities:
                #     iris_mobilities[new_polygon] = polygons[polygon]
                #     break

        # directory = '../data/infra_graphs/'+dep_name
        directory = "C:/Users/skian/Documents/ihu-covid/data/epci_graphs/"+dep_name
        if not os.path.exists(directory):
            os.makedirs(directory)
        f = open(directory+'/epci_FR_'+dat+'.csv','w')
        # for polygon in polygons:
        for iris_mobility in iris_mobilities:
            start_polygon = iris_mobility[0]
            end_polygon = iris_mobility[1]

            # pop = pop_irises[start_polygon]
            pop_dep = dep_pops[dep]
            pop = float(epcis_pops[start_polygon])/pop_dep
            # print(start_polygon+" "+end_polygon)
            # print(pop)
            # tup = (start_polygon, end_polygon)
            if pop!=0:
                f.write(start_polygon+","+end_polygon+","+str(float(iris_mobilities[iris_mobility]*pop))+"\n")
            # else:
            #     print(start_polygon+" "+end_polygon)
        f.close()


def dispatch_jobs_epci(data, job_number, dep_names, df_epci_pop, dep_pops, epcis_in, epcis_pops, polygons, neighs_d, dat):
# def dispatch_jobs_epci(data, job_number, dep_names, df_iris_pop, dep_pops, irises, irises_in, pop_irises, polygons, dat):
    total = len(data)
    # chunk_size = total / job_number
    # import ipdb; ipdb.set_trace()
    # slice = chunks(data, chunk_size)
    slice = np.array_split(list(data), job_number)
    jobs = []

    for i, s in enumerate(slice):
        j = multiprocessing.Process(target=do_job_epci, args=(i, s, dep_names, df_epci_pop, dep_pops, epcis_in, epcis_pops, polygons, neighs_d, dat))
        jobs.append(j)
    for j in jobs:
        j.start()


if __name__ == '__main__':
    # create_department_data()
    # create_vaccination_department_data()
    # extract_sci_data()
    # extract_departments_mobility_data()
    # extract_infra_dep_mobility_data()
    # get_france_nuts()

    # parse_iris_excel()
    create_infra_dep_dataset()
    final_infra_dataset()
    extract_infra_france_dep_mobility_data()

    # parse_epci_excel()
    # create_epci_dataset()
    # final_epci_dataset()
    # extract_epci_mobility_data()
