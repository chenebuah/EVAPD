## THIS SOURCE CODE DEVELOPS THE INPUT IMAGE FROM SCRATCH FOR ALL INVESTIGATED PEROVSKITE STOICHIOMETRIES
## THE CODE MUST BE EXECUTED ALONG-SIDE THE DATA (SPREADSHEETS) IN THE SAME DIRECTORY.


import pandas as pd
import numpy as np

abx3 = pd.read_csv('atoms_abx3.csv')
a2bbx6 = pd.read_csv('atoms_a2bbx6.csv')
aabbx6 = pd.read_csv('atoms_aabbx6.csv')

atomic_no = {
    "H":1, "He":2, "Li":3, "Be":4, "B":5, "C":6, "N":7, "O":8, "F":9, "Ne":10, "Na":11, "Mg":12, "Al":13, "Si":14, "P":15, "S":16, "Cl":17, "Ar":18, "K":19, "Ca":20,
    "Sc":21, "Ti":22, "V":23, "Cr":24, "Mn":25, "Fe":26, "Co":27, "Ni":28, "Cu":29, "Zn":30, "Ga":31, "Ge":32, "As":33, "Se":34, "Br":35, "Kr":36, "Rb":37, "Sr":38, 
    "Y":39, "Zr":40, "Nb":41, "Mo":42, "Tc":43, "Ru":44, "Rh":45, "Pd":46, "Ag":47, "Cd":48, "In":49, "Sn":50, "Sb":51, "Te":52, "I":53, "Xe":54, "Cs":55, "Ba":56, 
    "La":57, "Ce":58, "Pr":59, "Nd":60, "Pm":61, "Sm":62, "Eu":63, "Gd":64, "Tb":65, "Dy":66, "Ho":67, "Er":68, "Tm":69, "Yb":70, "Lu":71, "Hf":72, "Ta":73, "W":74, 
    "Re":75, "Os":76, "Ir":77, "Pt":78, "Au":79, "Hg":80, "Tl":81, "Pb":82, "Bi":83, "Po":84, "At":85, "Rn":86, "Fr":87, "Ra":88, "Ac":89, "Th":90, "Pa":91, "U":92,
    "Np":93, "Pu":94, "Am":95, "Cm":96, "Bk":97, "Cf":98, "Es":99, "Fm":100, "Md":101, "No":102, "Lr":103
    }

group_no = {
    "H":1, "He":18, "Li":1, "Be":2, "B":13, "C":14, "N":15, "O":16, "F":17, "Ne":18, "Na":1, "Mg":2, "Al":13, "Si":14, "P":15, "S":16, "Cl":17, "Ar":18, "K":1, "Ca":2,
    "Sc":3, "Ti":4, "V":5, "Cr":6, "Mn":7, "Fe":8, "Co":9, "Ni":10, "Cu":11, "Zn":12, "Ga":13, "Ge":14, "As":15, "Se":16, "Br":17, "Kr":18, "Rb":1, "Sr":2, 
    "Y":3, "Zr":4, "Nb":5, "Mo":6, "Tc":7, "Ru":8, "Rh":9, "Pd":10, "Ag":11, "Cd":12, "In":13, "Sn":14, "Sb":15, "Te":16, "I":17, "Xe":18, "Cs":1, "Ba":2, 
    "La":3, "Ce":4, "Pr":5, "Nd":6, "Pm":7, "Sm":8, "Eu":9, "Gd":10, "Tb":11, "Dy":12, "Ho":13, "Er":14, "Tm":15, "Yb":16, "Lu":17, "Hf":4, "Ta":5, "W":6, 
    "Re":7, "Os":8, "Ir":9, "Pt":10, "Au":11, "Hg":12, "Tl":13, "Pb":14, "Bi":15, "Po":16, "At":17, "Rn":18, "Fr":1, "Ra":2, "Ac":3, "Th":4, "Pa":5, "U":6,
    "Np":7, "Pu":8, "Am":9, "Cm":10, "Bk":11, "Cf":12, "Es":13, "Fm":14, "Md":15, "No":16, "Lr":17
    }

row_no = {
    "H":1, "He":1, "Li":2, "Be":2, "B":2, "C":2, "N":2, "O":2, "F":2, "Ne":2, "Na":3, "Mg":3, "Al":3, "Si":3, "P":3, "S":3, "Cl":3, "Ar":3, "K":4, "Ca":4,
    "Sc":4, "Ti":4, "V":4, "Cr":4, "Mn":4, "Fe":4, "Co":4, "Ni":4, "Cu":4, "Zn":4, "Ga":4, "Ge":4, "As":4, "Se":4, "Br":4, "Kr":4, "Rb":5, "Sr":5, 
    "Y":5, "Zr":5, "Nb":5, "Mo":5, "Tc":5, "Ru":5, "Rh":5, "Pd":5, "Ag":5, "Cd":5, "In":5, "Sn":5, "Sb":5, "Te":5, "I":5, "Xe":5, "Cs":6, "Ba":6, 
    "La":8, "Ce":8, "Pr":8, "Nd":8, "Pm":8, "Sm":8, "Eu":8, "Gd":8, "Tb":8, "Dy":8, "Ho":8, "Er":8, "Tm":8, "Yb":8, "Lu":8, "Hf":6, "Ta":6, "W":6, 
    "Re":6, "Os":6, "Ir":6, "Pt":6, "Au":6, "Hg":6, "Tl":6, "Pb":6, "Bi":6, "Po":6, "At":6, "Rn":6, "Fr":7, "Ra":7, "Ac":9, "Th":9, "Pa":9, "U":9,
    "Np":9, "Pu":9, "Am":9, "Cm":9, "Bk":9, "Cf":9, "Es":9, "Fm":9, "Md":9, "No":9, "Lr":9
    }

valence = {
    "H":1, "He":2, "Li":1, "Be":2, "B":3, "C":4, "N":5, "O":6, "F":7, "Ne":8, "Na":1, "Mg":2, "Al":3, "Si":4, "P":5, "S":6, "Cl":7, "Ar":8, "K":1, "Ca":2,
    "Sc":2, "Ti":2, "V":2, "Cr":1, "Mn":2, "Fe":2, "Co":2, "Ni":2, "Cu":1, "Zn":2, "Ga":3, "Ge":4, "As":5, "Se":6, "Br":7, "Kr":8, "Rb":1, "Sr":2, 
    "Y":2, "Zr":2, "Nb":1, "Mo":1, "Tc":2, "Ru":1, "Rh":1, "Pd":9, "Ag":1, "Cd":2, "In":3, "Sn":4, "Sb":5, "Te":6, "I":7, "Xe":8, "Cs":1, "Ba":2, 
    "La":2, "Ce":2, "Pr":2, "Nd":2, "Pm":2, "Sm":2, "Eu":2, "Gd":2, "Tb":2, "Dy":2, "Ho":2, "Er":2, "Tm":2, "Yb":2, "Lu":2, "Hf":2, "Ta":2, "W":2, 
    "Re":2, "Os":2, "Ir":2, "Pt":1, "Au":1, "Hg":2, "Tl":3, "Pb":4, "Bi":5, "Po":6, "At":7, "Rn":8, "Fr":1, "Ra":2, "Ac":2, "Th":2, "Pa":2, "U":2,
    "Np":2, "Pu":2, "Am":2, "Cm":2, "Bk":2, "Cf":2, "Es":2, "Fm":2, "Md":2, "No":2, "Lr":3
    }

block = {
    "H":1, "He":1, "Li":1, "Be":1, "B":2, "C":2, "N":2, "O":2, "F":2, "Ne":2, "Na":1, "Mg":1, "Al":2, "Si":2, "P":2, "S":2, "Cl":2, "Ar":2, "K":1, "Ca":1,
    "Sc":3, "Ti":3, "V":3, "Cr":3, "Mn":3, "Fe":3, "Co":3, "Ni":3, "Cu":3, "Zn":3, "Ga":2, "Ge":2, "As":2, "Se":2, "Br":2, "Kr":2, "Rb":1, "Sr":1, 
    "Y":3, "Zr":3, "Nb":3, "Mo":3, "Tc":3, "Ru":3, "Rh":3, "Pd":3, "Ag":3, "Cd":3, "In":2, "Sn":2, "Sb":2, "Te":2, "I":2, "Xe":2, "Cs":1, "Ba":1, 
    "La":4, "Ce":4, "Pr":4, "Nd":4, "Pm":4, "Sm":4, "Eu":4, "Gd":4, "Tb":4, "Dy":4, "Ho":4, "Er":4, "Tm":4, "Yb":4, "Lu":3, "Hf":3, "Ta":3, "W":3, 
    "Re":3, "Os":3, "Ir":3, "Pt":3, "Au":3, "Hg":3, "Tl":2, "Pb":2, "Bi":2, "Po":2, "At":2, "Rn":2, "Fr":1, "Ra":1, "Ac":4, "Th":4, "Pa":4, "U":4,
    "Np":4, "Pu":4, "Am":4, "Cm":4, "Bk":4, "Cf":4, "Es":4, "Fm":4, "Md":4, "No":4, "Lr":3
    }

electronegativity = {
    'H':2.2, 'He':'n.a', 'Li':0.98, 'Be':	1.57, 'B':	2.04, 'C':2.55,'N':	3.04,'O':	3.44,'F':	3.98,'Ne':	"n.a",'Na':	0.93,'Mg':	1.31,'Al':	1.61,'Si':	1.9,
    'P':	2.19,'S':	2.58,'Cl':	3.16,'Ar':	"n.a", 'K':	0.82, 'Ca':	1, 'Sc':	1.36, 'Ti':	1.54, 'V':	1.63,'Cr':	1.66,'Mn':	1.55,'Fe':	1.83,'Co':	1.88,'Ni':	1.91,
    'Cu':	1.9, 'Zn':	1.65, 'Ga':	1.81, 'Ge':	2.01,'As':	2.18,'Se':	2.55,'Br':	2.96,'Kr':	"n.a",'Rb':	0.82,'Sr':	0.95,'Y':	1.22,'Zr':	1.33,'Nb':	1.6,'Mo':	2.16,
    'Tc':	2.1,'Ru':	2.2,'Rh':	2.28,'Pd':	2.2,'Ag':	1.93,'Cd':	1.69,'In':	1.78,'Sn':	1.96,'Sb':	2.05,'Te':	2.1,'I':	2.66,'Xe':	2.6,'Cs':	0.79,'Ba':	0.89,
    'La':	1.1,'Ce':	1.12,'Pr':	1.13,'Nd':	1.14,'Pm':	1.13,'Sm':	1.17,'Eu':	1.2,'Gd':	1.2,'Tb':	1.1,'Dy':	1.22,'Ho':	1.23,'Er':	1.24,'Tm':	1.25,'Yb':	1.1,
    'Lu':	1,'Hf':	1.3,'Ta':	1.5,'W':	1.7,'Re':	1.9,'Os':	2.2,'Ir':	2.2,'Pt':	2.2,'Au':	2.4,'Hg':	1.9,'Tl':	1.8,'Pb':	1.8,'Bi':	1.9,'Po':	2,'At':	2.2,'Rn':	2.2,'Fr':	0.7,
    'Ra':	0.9,'Ac':	1.1,'Th':	1.3,'Pa':	1.5,'U':	1.7,'Np':	1.3,'Pu':	1.3,'Am':	1.3,'Cm':	1.3,'Bk':	1.3,'Cf':	1.3,'Es':	1.3,'Fm':	1.3,'Md':	1.3,'No':	1.3,'Lr':	1.3
    }

covalent_radius = {
    'H':	0.31,'He':	0.28,'Li':	1.28,'Be':	0.96,'B':	0.84,'C':	0.76,'N':	0.71,'O':	0.66,'F':	0.57,'Ne':	0.58,'Na':	1.66,'Mg':	1.41,'Al':	1.21,'Si':	1.11,'P':	1.07,
    'S':	1.05,'Cl':	1.02,'Ar':	1.06,'K':	2.03,'Ca':	1.76,'Sc':	1.7,'Ti':	1.6,'V':	1.53,'Cr':	1.39,'Mn':	1.39,'Fe':	1.32,'Co':	1.26,'Ni':	1.24,'Cu':	1.32,'Zn':	1.22,
    'Ga':	1.22,'Ge':	1.2,'As':	1.19,'Se':	1.2,'Br':	1.2,'Kr':	1.16,'Rb':	2.2,'Sr':	1.95,'Y':	1.9,'Zr':	1.75,'Nb':	1.64,'Mo':	1.54,'Tc':	1.47,'Ru':	1.46,'Rh':	1.42,
    'Pd':	1.39,'Ag':	1.45,'Cd':	1.44,'In':	1.42,'Sn':	1.39,'Sb':	1.39,'Te':	1.38,'I':	1.39,'Xe':	1.4,'Cs':	2.44,'Ba':	2.15,'La':	2.07,'Ce':	2.04,'Pr':	2.03,'Nd':	2.01,
    'Pm':	1.99,'Sm':	1.98,'Eu':	1.98,'Gd':	1.96,'Tb':	1.94,'Dy':	1.92,'Ho':	1.92,'Er':	1.89,'Tm':	1.9,'Yb':	1.87,'Lu':	1.87,'Hf':	1.75,'Ta':	1.7,'W':	1.62,'Re':	1.51,
    'Os':	1.44,'Ir':	1.41,'Pt':	1.36,'Au':	1.36,'Hg':	1.32,'Tl':	1.45,'Pb':	1.46,'Bi':	1.48,'Po':	1.4,'At':	1.5,'Rn':	1.5,'Fr':	2.6,'Ra':	2.21,'Ac':	2.15,'Th':	2.06,
    'Pa':	2,'U':	1.96,'Np':	1.9,'Pu':	1.87,'Am':	1.8,'Cm':	1.69,'Bk':	'n.a','Cf':	'n.a','Es':	'n.a','Fm':	'n.a','Md':	'n.a','No':	'n.a','Lr':	'n.a'
    }

ionization_energy = {
    'H':	13.59844,'He':	24.58741,'Li':	5.39172,'Be':	9.3227,'B':	8.29803,'C':	11.2603,'N':	14.53414,'O':	13.61806,'F':	17.42282,'Ne':	21.5646,'Na':	5.13908,'Mg':	7.64624,
    'Al':	5.98577,'Si':	8.15169,'P':	10.48669,'S':	10.36001,'Cl':	12.96764,'Ar':	15.75962,'K':	4.34066,'Ca':	6.11316,'Sc':	6.5615,'Ti':	6.8281,'V':	6.7462,'Cr':	6.7665,
    'Mn':	7.43402,'Fe':	7.9024,'Co':	7.881,'Ni':	7.6398,'Cu':	7.72638,'Zn':	9.3942,'Ga':	5.9993,'Ge':	7.8994,'As':	9.7886,'Se':	9.75238,'Br':	11.81381,'Kr':	13.99961,
    'Rb':	4.17713,'Sr':	5.6949,'Y':	6.2171,'Zr':	6.6339,'Nb':	6.75885,'Mo':	7.09243,'Tc':	7.28,'Ru':	7.3605,'Rh':	7.4589,'Pd':	8.3369,'Ag':	7.5762,'Cd':	8.9938,'In':	5.78636,
    'Sn':	7.3439,'Sb':	8.6084,'Te':	9.0096,'I':	10.45126,'Xe':	12.1298,'Cs':	3.8939,'Ba':	5.2117,'La':	5.5769,'Ce':	5.5387,'Pr':	5.473,'Nd':	5.525,'Pm':	5.582,'Sm':	5.6436,
    'Eu':	5.6704,'Gd':	6.1501,'Tb':	5.8638,'Dy':	5.9389,'Ho':	6.0215,'Er':	6.1077,'Tm':	6.18431,'Yb':	6.25416,'Lu':	5.4259,'Hf':	6.82507,'Ta':	7.5496,'W':	7.864,'Re':	7.8335,
    'Os':	8.4382,'Ir':	8.967,'Pt':	8.9587,'Au':	9.2255,'Hg':	10.4375,'Tl':	6.1082,'Pb':	7.41666,'Bi':	7.2856,'Po':	8.417,'At':	"n.a",'Rn':	10.7485,'Fr':	4.0727,'Ra':	5.2784,
    'Ac':	5.17,'Th':	6.3067,'Pa':	5.89,'U':	6.19405,'Np':	6.2657,'Pu':	6.0262,'Am':	5.9738,'Cm':	5.9915,'Bk':	6.1979,'Cf':	6.2817,'Es':	6.42,'Fm':	6.5,'Md':	6.58,'No':	6.65,
    'Lr':	4.9
    }

electron_affinity = {
    'H':	0.754195,'He':	-0.5182,'Li':	0.618049,'Be':	-0.5182,'B':	0.279723,'C':	1.262119,'N':	-0.0725,'O':	1.4611096,'F':	3.4011895,'Ne':	-1.2437,'Na':	0.547926,'Mg':	-0.4146,
    'Al':	0.43283,'Si':	1.389522,'P':	0.7465,'S':	2.077103,'Cl':	3.612724,'Ar':	-0.995,'K':	0.50147,'Ca':	0.02455,'Sc':	0.188,'Ti':	0.079,'V':	0.525,'Cr':	0.666,'Mn':	-0.5,
    'Fe':	0.151,'Co':	0.662,'Ni':	1.156,'Cu':	1.235,'Zn':	-0.6219,'Ga':	0.43,'Ge':	1.232712,'As':	0.814,'Se':	2.02067,'Br':	3.363588,'Kr':	-1,'Rb':	0.48592,'Sr':	0.048,'Y':	0.307,
    'Zr':	0.426,'Nb':	0.8933,'Mo':	0.748,'Tc':	0.55,'Ru':	1.05,'Rh':	1.137,'Pd':	0.562,'Ag':	1.302,'Cd':	-0.7255,'In':	0.3,'Sn':	1.112067,'Sb':	1.046,'Te':	1.9708,'I':	3.059037,
    'Xe':	-0.8291,'Cs':	0.471626,'Ba':	0.14462,'La':	0.47,'Ce':	0.57,'Pr':	0.964,'Nd':	0.09749,'Pm':	0.124,'Sm':	0.166,'Eu':	0.114,'Gd':	0.135,'Tb':	1.161,'Dy':	0.352,'Ho':	0.342,
    'Er':	0.311,'Tm':	1.03,'Yb':	-0.02,'Lu':	0.34,'Hf':	0.176,'Ta':	0.322,'W':	0.815,'Re':	0.15,'Os':	1.1,'Ir':	1.5638,'Pt':	2.128,'Au':	2.30863,'Hg':	-0.5182,'Tl':	0.2,'Pb':	0.364,
    'Bi':	0.946,'Po':	1.9,'At':	2.8,'Rn':	-0.7255,'Fr':	0.46,'Ra':	0.1,'Ac':	0.35,'Th':	0.60769,'Pa':	0.57,'U':	0.31497,'Np':	0.477,'Pu':	-0.5,'Am':	0.1,'Cm':	0.28,'Bk':	-1.72,
    'Cf':	-1.01,'Es':	-0.3,'Fm':	0.35,'Md':	0.98,'No':	-2.33,'Lr':	-0.31
    }

molar_volume = {
    'H':	11.42,'He':	21,'Li':	13.02,'Be':	4.85,'B':	4.39,'C':	5.29,'N':	13.54,'O':	17.36,'F':	11.2,'Ne':	13.23,'Na':	23.78,'Mg':	14,'Al':	10,'Si':	12.06,'P':	17.02,'S':	15.53,
    'Cl':	17.39,'Ar':	22.56,'K':	45.94,'Ca':	26.2,'Sc':	15,'Ti':	10.64,'V':	8.32,'Cr':	7.23,'Mn':	7.35,'Fe':	7.09,'Co':	6.67,'Ni':	6.59,'Cu':	7.11,'Zn':	9.16,'Ga':	11.8,
    'Ge':	13.63,'As':	12.95,'Se':	16.42,'Br':	19.78,'Kr':	27.99,'Rb':	55.76,'Sr':	33.94,'Y':	19.88,'Zr':	14.02,'Nb':	10.83,'Mo':	9.38,'Tc':	8.63,'Ru':	8.17,'Rh':	8.28,'Pd':	8.56,
    'Ag':	10.27,'Cd':	13,'In':	15.76,'Sn':	16.29,'Sb':	18.19,'Te':	20.46,'I':	25.72,'Xe':	35.92,'Cs':	70.94,'Ba':	38.16,'La':	22.39,'Ce':	20.69,'Pr':	20.8,'Nd':	20.59,'Pm':	20.23,
    'Sm':	19.98,'Eu':	28.97,'Gd':	19.9,'Tb':	19.3,'Dy':	19.01,'Ho':	18.74,'Er':	18.46,'Tm':	19.1,'Yb':	24.84,'Lu':	17.78,'Hf':	13.44,'Ta':	10.85,'W':	9.47,'Re':	8.86,'Os':	8.42,
    'Ir':	8.52,'Pt':	9.09,'Au':	10.21,'Hg':	14.09,'Tl':	17.22,'Pb':	18.26,'Bi':	21.31,'Po':	22.97,'At':	"n.a",'Rn':	50.5,'Fr':	"n.a",'Ra':	41.09,'Ac':	22.55,'Th':	19.8,'Pa':	15.18,
    'U':	12.49,'Np':	11.59,'Pu':	12.29,'Am':	17.63,'Cm':	18.05,'Bk':	16.84,'Cf':	16.5,'Es':	28.52,'Fm':	"n.a",'Md':	"n.a",'No':	"n.a",'Lr':	"n.a"
    }

average_ionic_radius = {
    'H':	0,'He':	0,'Li':	0.9,'Be':	0.59,'B':	0.41,'C':	0.3,'N':	0.63,'O':	1.26,'F':	0.705,'Ne':	0,'Na':	1.16,'Mg':	0.86,'Al':	0.675,'Si':	0.54,'P':	0.55,'S':	0.88,'Cl':	0.78,
    'Ar':	0,'K':	1.52,'Ca':	1.14,'Sc':	0.885,'Ti':	0.852,'V':	0.777,'Cr':	0.94,'Mn':	0.648,'Fe':	0.853,'Co':	0.768,'Ni':	0.74,'Cu':	0.82,'Zn':	0.88,'Ga':	0.76,'Ge':	0.77,
    'As':	0.66,'Se':	1.013,'Br':	0.883,'Kr':	0,'Rb':	1.66,'Sr':	1.32,'Y':	1.04,'Zr':	0.86,'Nb':	0.82,'Mo':	0.775,'Tc':	0.742,'Ru':	0.661,'Rh':	0.745,'Pd':	0.846,'Ag':	1.087,
    'Cd':	1.09,'In':	0.94,'Sn':	0.83,'Sb':	0.83,'Te':	1.293,'I':	1.273,'Xe':	0.62,'Cs':	1.81,'Ba':	1.49,'La':	1.172,'Ce':	1.08,'Pr':	1.06,'Nd':	1.276,'Pm':	1.11,'Sm':	1.229,
    'Eu':	1.199,'Gd':	1.075,'Tb':	0.982,'Dy':	1.131,'Ho':	1.041,'Er':	1.03,'Tm':	1.095,'Yb':	1.084,'Lu':	1.001,'Hf':	0.85,'Ta':	0.82,'W':	0.767,'Re':	0.712,'Os':	0.673,'Ir':	0.765,
    'Pt':	0.805,'Au':	1.07,'Hg':	1.245,'Tl':	1.333,'Pb':	1.123,'Bi':	1.035,'Po':	0.945,'At':	0.76,'Rn':	0,'Fr':	1.94,'Ra':	1.62,'Ac':	1.26,'Th':	1.08,'Pa':	1.04,'U':	0.991,'Np':	1,
    'Pu':	0.967,'Am':	1.168,'Cm':	1.05,'Bk':	1.035,'Cf':	1.026,'Es':	0,'Fm':	0,'Md':	0,'No':	0,'Lr':	0
    }

polarizability = {
    'H':	0.666793,'He':	0.204956,'Li':	24.3,'Be':	5.6,'B':	3.03,'C':	1.76,'N':	1.1,'O':	0.802,'F':	0.557,'Ne':	0.3956,'Na':	24.11,'Mg':	10.6,'Al':	6.8,'Si':	5.38,'P':	3.63,
    'S':	2.9,'Cl':	2.18,'Ar':	1.6411,'K':	43.4,'Ca':	22.8,'Sc':	17.8,'Ti':	14.6,'V':	12.4,'Cr':	11.6,'Mn':	9.4,'Fe':	8.4,'Co':	7.5,'Ni':	6.8,'Cu':	6.2,'Zn':	5.75,'Ga':	8.12,
    'Ge':	6.07,'As':	4.31,'Se':	3.77,'Br':	3.05,'Kr':	2.4844,'Rb':	47.3,'Sr':	27.6,'Y':	22.7,'Zr':	17.9,'Nb':	15.7,'Mo':	12.8,'Tc':	11.4,'Ru':	9.6,'Rh':	8.6,'Pd':	4.8,
    'Ag':	7.2,'Cd':	7.36,'In':	10.2,'Sn':	7.7,'Sb':	6.6,'Te':	5.5,'I':	5.35,'Xe':	4.044,'Cs':	59.42,'Ba':	39.7,'La':	31.1,'Ce':	29.6,'Pr':	28.2,'Nd':	31.4,'Pm':	30.1,
    'Sm':	28.8,'Eu':	27.7,'Gd':	23.5,'Tb':	25.5,'Dy':	24.5,'Ho':	23.6,'Er':	22.7,'Tm':	21.8,'Yb':	21,'Lu':	21.9,'Hf':	16.2,'Ta':	13.1,'W':	11.1,'Re':	9.7,'Os':	8.5,
    'Ir':	7.6,'Pt':	6.5,'Au':	5.8,'Hg':	5.02,'Tl':	7.6,'Pb':	6.8,'Bi':	7.4,'Po':	6.8,'At':	6,'Rn':	5.3,'Fr':	47.1,'Ra':	38.3,'Ac':	32.1,'Th':	32.1,'Pa':	25.4,'U':	24.9,'Np':	24.8,
    'Pu':	24.5,'Am':	23.3,'Cm':	23,'Bk':	22.7,'Cf':	20.5,'Es':	19.7,'Fm':	23.8,'Md':	18.2,'No':	17.5,'Lr':	"n.a."
    }

specific_heat = {
    'H':	14.304,'He':	5.193,'Li':	3.582,'Be':	1.825,'B':	1.026,'C':	0.709,'N':	1.04,'O':	0.918,'F':	0.824,'Ne':	1.03,'Na':	1.228,'Mg':	1.023,'Al':	0.897,'Si':	0.705,'P':	0.769,
    'S':	0.71,'Cl':	0.479,'Ar':	0.52,'K':	0.757,'Ca':	0.647,'Sc':	0.568,'Ti':	0.523,'V':	0.489,'Cr':	0.449,'Mn':	0.479,'Fe':	0.449,'Co':	0.421,'Ni':	0.444,'Cu':	0.385,'Zn':	0.388,
    'Ga':	0.371,'Ge':	0.32,'As':	0.329,'Se':	0.321,'Br':	0.474,'Kr':	0.248,'Rb':	0.363,'Sr':	0.301,'Y':	0.298,'Zr':	0.278,'Nb':	0.265,'Mo':	0.251,'Tc':	0.063,'Ru':	0.238,'Rh':	0.243,
    'Pd':	0.244,'Ag':	0.235,'Cd':	0.232,'In':	0.233,'Sn':	0.228,'Sb':	0.207,'Te':	0.202,'I':	0.214,'Xe':	0.158,'Cs':	0.242,'Ba':	0.204,'La':	0.195,'Ce':	0.192,'Pr':	0.193,'Nd':	0.19,
    'Pm':	0.18,'Sm':	0.197,'Eu':	0.182,'Gd':	0.236,'Tb':	0.182,'Dy':	0.17,'Ho':	0.165,'Er':	0.168,'Tm':	0.16,'Yb':	0.155,'Lu':	0.154,'Hf':	0.144,'Ta':	0.14,'W':	0.132,'Re':	0.137,
    'Os':	0.13,'Ir':	0.131,'Pt':	0.133,'Au':	0.129,'Hg':	0.14,'Tl':	0.129,'Pb':	0.129,'Bi':	0.122,'Po':	"n.a",'At':	"n.a",'Rn':	0.094,'Fr':	"n.a",'Ra':	0.092,'Ac':	0.12,'Th':	0.113,
    'Pa':	0.0991,'U':	0.116,'Np':	0.12,'Pu':	0.13,'Am':	"n.a",'Cm':	"n.a",'Bk':	"n.a",'Cf':	"n.a",'Es':	"n.a",'Fm':	"n.a",'Md':	"n.a",'No':	"n.a",'Lr':	"n.a"
    }

thermal_conductivity = {
    'H':	0.1805,'He':	0.1513,'Li':	85,'Be':	190,'B':	27,'C':	140,'N':	0.02583,'O':	0.02658,'F':	0.0277,'Ne':	0.0491,'Na':	140,'Mg':	160,'Al':	235,'Si':	150,'P':	0.236,
    'S':	0.205,'Cl':	0.0089,'Ar':	0.01772,'K':	100,'Ca':	200,'Sc':	16,'Ti':	22,'V':	31,'Cr':	94,'Mn':	7.8,'Fe':	80,'Co':	100,'Ni':	91,'Cu':	400,'Zn':	120,'Ga':	29,'Ge':	60,
    'As':	50,'Se':	2.04,'Br':	0.12,'Kr':	0.00943,'Rb':	58,'Sr':	35,'Y':	17,'Zr':	23,'Nb':	54,'Mo':	139,'Tc':	51,'Ru':	120,'Rh':	150,'Pd':	72,'Ag':	430,'Cd':	97,'In':	82,
    'Sn':	67,'Sb':	24,'Te':	3,'I':	0.449,'Xe':	0.00565,'Cs':	36,'Ba':	18,'La':	13,'Ce':	11,'Pr':	13,'Nd':	17,'Pm':	17.9,'Sm':	13,'Eu':	14,'Gd':	11,'Tb':	11,'Dy':	11,
    'Ho':	16,'Er':	15,'Tm':	17,'Yb':	39,'Lu':	16,'Hf':	23,'Ta':	57,'W':	170,'Re':	48,'Os':	88,'Ir':	150,'Pt':	72,'Au':	320,'Hg':	8.3,'Tl':	46,'Pb':	35,'Bi':	8,'Po':	20,
    'At':	2,'Rn':	0.00361,'Fr':	77,'Ra':	19,'Ac':	12,'Th':	54,'Pa':	47,'U':	27,'Np':	6,'Pu':	6,'Am':	10,'Cm':	8.8,'Bk':	10,'Cf':	10,'Es':	10,'Fm':	10,'Md':	10,'No':	10,
    'Lr':	10
    }

# Atomic Number
abx3_datasize = abx3.shape[0]
a2bbx6_datasize = a2bbx6.shape[0]
aabbx6_datasize = aabbx6.shape[0]

data1=[]
for i in range (0, abx3_datasize):
  z=atomic_no[abx3.iloc[(i,0)]], atomic_no[abx3.iloc[(i,1)]], atomic_no[abx3.iloc[(i,2)]]
  data1.append(z)

Z_abx3=(np.array(data1))-1

data1=[]
for i in range (0, a2bbx6_datasize):
  z=atomic_no[a2bbx6.iloc[(i,0)]], atomic_no[a2bbx6.iloc[(i,1)]], atomic_no[a2bbx6.iloc[(i,2)]], atomic_no[a2bbx6.iloc[(i,3)]]
  data1.append(z)

Z_a2bbx6=(np.array(data1))-1

data1=[]
for i in range (0, aabbx6_datasize):
  z=atomic_no[aabbx6.iloc[(i,0)]], atomic_no[aabbx6.iloc[(i,1)]], atomic_no[aabbx6.iloc[(i,2)]], atomic_no[aabbx6.iloc[(i,3)]], atomic_no[aabbx6.iloc[(i,4)]]
  data1.append(z)

Z_aabbx6=(np.array(data1))-1

# Group Number
data2=[]
for i in range (0, abx3_datasize):
  gn=group_no[abx3.iloc[(i,0)]], group_no[abx3.iloc[(i,1)]], group_no[abx3.iloc[(i,2)]]
  data2.append(gn)

GN_abx3=(np.array(data2))-1

data2=[]
for i in range (0, a2bbx6_datasize):
  gn=group_no[a2bbx6.iloc[(i,0)]], group_no[a2bbx6.iloc[(i,1)]], group_no[a2bbx6.iloc[(i,2)]], group_no[a2bbx6.iloc[(i,3)]]
  data2.append(gn)

GN_a2bbx6=(np.array(data2))-1

data2=[]
for i in range (0, aabbx6_datasize):
  gn=group_no[aabbx6.iloc[(i,0)]], group_no[aabbx6.iloc[(i,1)]], group_no[aabbx6.iloc[(i,2)]], group_no[aabbx6.iloc[(i,3)]], group_no[aabbx6.iloc[(i,4)]]
  data2.append(gn)

GN_aabbx6=(np.array(data2))-1

# Row Number

data3=[]
for i in range (0, abx3_datasize):
  rn=row_no[abx3.iloc[(i,0)]], row_no[abx3.iloc[(i,1)]], row_no[abx3.iloc[(i,2)]]
  data3.append(rn)

RN_abx3=(np.array(data3))-1

data3=[]
for i in range (0, a2bbx6_datasize):
  rn=row_no[a2bbx6.iloc[(i,0)]], row_no[a2bbx6.iloc[(i,1)]], row_no[a2bbx6.iloc[(i,2)]], row_no[a2bbx6.iloc[(i,3)]]
  data3.append(rn)

RN_a2bbx6=(np.array(data3))-1

data3=[]
for i in range (0, aabbx6_datasize):
  rn=row_no[aabbx6.iloc[(i,0)]], row_no[aabbx6.iloc[(i,1)]], row_no[aabbx6.iloc[(i,2)]], row_no[aabbx6.iloc[(i,3)]], row_no[aabbx6.iloc[(i,4)]]
  data3.append(rn)

RN_aabbx6=(np.array(data3))-1

# Valence

data4=[]
for i in range (0, abx3_datasize):
  vl=valence[abx3.iloc[(i,0)]], valence[abx3.iloc[(i,1)]], valence[abx3.iloc[(i,2)]]
  data4.append(vl)

VL_abx3=(np.array(data4))-1

data4=[]
for i in range (0, a2bbx6_datasize):
  vl=valence[a2bbx6.iloc[(i,0)]], valence[a2bbx6.iloc[(i,1)]], valence[a2bbx6.iloc[(i,2)]], valence[a2bbx6.iloc[(i,3)]]
  data4.append(vl)

VL_a2bbx6=(np.array(data4))-1

data4=[]
for i in range (0, aabbx6_datasize):
  vl=valence[aabbx6.iloc[(i,0)]], valence[aabbx6.iloc[(i,1)]], valence[aabbx6.iloc[(i,2)]], valence[aabbx6.iloc[(i,3)]], valence[aabbx6.iloc[(i,4)]]
  data4.append(vl)

VL_aabbx6=(np.array(data4))-1

# Block

data5=[]
for i in range (0, abx3_datasize):
  bk=block[abx3.iloc[(i,0)]], block[abx3.iloc[(i,1)]], block[abx3.iloc[(i,2)]]
  data5.append(bk)

BK_abx3=(np.array(data5))-1

data5=[]
for i in range (0, a2bbx6_datasize):
  bk=block[a2bbx6.iloc[(i,0)]], block[a2bbx6.iloc[(i,1)]], block[a2bbx6.iloc[(i,2)]], block[a2bbx6.iloc[(i,3)]]
  data5.append(bk)

BK_a2bbx6=(np.array(data5))-1

data5=[]
for i in range (0, aabbx6_datasize):
  bk=block[aabbx6.iloc[(i,0)]], block[aabbx6.iloc[(i,1)]], block[aabbx6.iloc[(i,2)]], block[aabbx6.iloc[(i,3)]], block[aabbx6.iloc[(i,4)]]
  data5.append(bk)

BK_aabbx6=(np.array(data5))-1

# Electronegativity

data6=[]
for i in range (0, abx3_datasize):
  x=electronegativity[abx3.iloc[(i,0)]], electronegativity[abx3.iloc[(i,1)]], electronegativity[abx3.iloc[(i,2)]]
  data6.append(x)

X_abx3=(np.array(data6))

data6=[]
for i in range (0, a2bbx6_datasize):
  x=electronegativity[a2bbx6.iloc[(i,0)]], electronegativity[a2bbx6.iloc[(i,1)]], electronegativity[a2bbx6.iloc[(i,2)]], electronegativity[a2bbx6.iloc[(i,3)]]
  data6.append(x)

X_a2bbx6=(np.array(data6))

data6=[]
for i in range (0, aabbx6_datasize):
  x=(electronegativity[aabbx6.iloc[(i,0)]], electronegativity[aabbx6.iloc[(i,1)]], electronegativity[aabbx6.iloc[(i,2)]], 
     electronegativity[aabbx6.iloc[(i,3)]], electronegativity[aabbx6.iloc[(i,4)]])
  data6.append(x)

X_aabbx6=(np.array(data6))

# Covalent Radius

data7=[]
for i in range (0, abx3_datasize):
  cr=covalent_radius[abx3.iloc[(i,0)]], covalent_radius[abx3.iloc[(i,1)]], covalent_radius[abx3.iloc[(i,2)]]
  data7.append(cr)

CR_abx3=(np.array(data7))

data7=[]
for i in range (0, a2bbx6_datasize):
  cr=covalent_radius[a2bbx6.iloc[(i,0)]], covalent_radius[a2bbx6.iloc[(i,1)]], covalent_radius[a2bbx6.iloc[(i,2)]], covalent_radius[a2bbx6.iloc[(i,3)]]
  data7.append(cr)

CR_a2bbx6=(np.array(data7))

data7=[]
for i in range (0, aabbx6_datasize):
  cr=covalent_radius[aabbx6.iloc[(i,0)]], covalent_radius[aabbx6.iloc[(i,1)]], covalent_radius[aabbx6.iloc[(i,2)]], covalent_radius[aabbx6.iloc[(i,3)]], covalent_radius[aabbx6.iloc[(i,4)]]
  data7.append(cr)

CR_aabbx6=(np.array(data7))

# Ionization Energy

data8=[]
for i in range (0, abx3_datasize):
  ie=ionization_energy[abx3.iloc[(i,0)]], ionization_energy[abx3.iloc[(i,1)]], ionization_energy[abx3.iloc[(i,2)]]
  data8.append(ie)

IE_abx3=(np.array(data8))

data8=[]
for i in range (0, a2bbx6_datasize):
  ie=ionization_energy[a2bbx6.iloc[(i,0)]], ionization_energy[a2bbx6.iloc[(i,1)]], ionization_energy[a2bbx6.iloc[(i,2)]], ionization_energy[a2bbx6.iloc[(i,3)]]
  data8.append(ie)

IE_a2bbx6=(np.array(data8))

data8=[]
for i in range (0, aabbx6_datasize):
  ie=(ionization_energy[aabbx6.iloc[(i,0)]], ionization_energy[aabbx6.iloc[(i,1)]], ionization_energy[aabbx6.iloc[(i,2)]], 
      ionization_energy[aabbx6.iloc[(i,3)]], ionization_energy[aabbx6.iloc[(i,4)]])
  data8.append(ie)

IE_aabbx6=(np.array(data8))

# Electron Affinity

data9=[]
for i in range (0, abx3_datasize):
  ea=electron_affinity[abx3.iloc[(i,0)]], electron_affinity[abx3.iloc[(i,1)]], electron_affinity[abx3.iloc[(i,2)]]
  data9.append(ea)

EA_abx3=(np.array(data9))

data9=[]
for i in range (0, a2bbx6_datasize):
  ea=electron_affinity[a2bbx6.iloc[(i,0)]], electron_affinity[a2bbx6.iloc[(i,1)]], electron_affinity[a2bbx6.iloc[(i,2)]], electron_affinity[a2bbx6.iloc[(i,3)]]
  data9.append(ea)

EA_a2bbx6=(np.array(data9))

data9=[]
for i in range (0, aabbx6_datasize):
  ea=(electron_affinity[aabbx6.iloc[(i,0)]], electron_affinity[aabbx6.iloc[(i,1)]], electron_affinity[aabbx6.iloc[(i,2)]], 
      electron_affinity[aabbx6.iloc[(i,3)]], electron_affinity[aabbx6.iloc[(i,4)]])
  data9.append(ea)

EA_aabbx6=(np.array(data9))

# Molar Volume

data10=[]
for i in range (0, abx3_datasize):
  mv=molar_volume[abx3.iloc[(i,0)]], molar_volume[abx3.iloc[(i,1)]], molar_volume[abx3.iloc[(i,2)]]
  data10.append(mv)

MV_abx3=(np.array(data10))

data10=[]
for i in range (0, a2bbx6_datasize):
  mv=molar_volume[a2bbx6.iloc[(i,0)]], molar_volume[a2bbx6.iloc[(i,1)]], molar_volume[a2bbx6.iloc[(i,2)]], molar_volume[a2bbx6.iloc[(i,3)]]
  data10.append(mv)

MV_a2bbx6=(np.array(data10))

data10=[]
for i in range (0, aabbx6_datasize):
  mv=molar_volume[aabbx6.iloc[(i,0)]], molar_volume[aabbx6.iloc[(i,1)]], molar_volume[aabbx6.iloc[(i,2)]], molar_volume[aabbx6.iloc[(i,3)]], molar_volume[aabbx6.iloc[(i,4)]]
  data10.append(mv)

MV_aabbx6=(np.array(data10))

# Average Ionic Radius

data11=[]
for i in range (0, abx3_datasize):
  ir=average_ionic_radius[abx3.iloc[(i,0)]], average_ionic_radius[abx3.iloc[(i,1)]], average_ionic_radius[abx3.iloc[(i,2)]]
  data11.append(ir)

IR_abx3=(np.array(data11))

data11=[]
for i in range (0, a2bbx6_datasize):
  ir=average_ionic_radius[a2bbx6.iloc[(i,0)]], average_ionic_radius[a2bbx6.iloc[(i,1)]], average_ionic_radius[a2bbx6.iloc[(i,2)]], average_ionic_radius[a2bbx6.iloc[(i,3)]]
  data11.append(ir)

IR_a2bbx6=(np.array(data11))

data11=[]
for i in range (0, aabbx6_datasize):
  ir=(average_ionic_radius[aabbx6.iloc[(i,0)]], average_ionic_radius[aabbx6.iloc[(i,1)]], average_ionic_radius[aabbx6.iloc[(i,2)]], 
      average_ionic_radius[aabbx6.iloc[(i,3)]], average_ionic_radius[aabbx6.iloc[(i,4)]])
  data11.append(ir)

IR_aabbx6=(np.array(data11))

# Polarizability

data12=[]
for i in range (0, abx3_datasize):
  pz=polarizability[abx3.iloc[(i,0)]], polarizability[abx3.iloc[(i,1)]], polarizability[abx3.iloc[(i,2)]]
  data12.append(pz)

PZ_abx3=(np.array(data12))

data12=[]

for i in range (0, a2bbx6_datasize):
  pz=polarizability[a2bbx6.iloc[(i,0)]], polarizability[a2bbx6.iloc[(i,1)]], polarizability[a2bbx6.iloc[(i,2)]], polarizability[a2bbx6.iloc[(i,3)]]
  data12.append(pz)

PZ_a2bbx6=(np.array(data12))

data12=[]

for i in range (0, aabbx6_datasize):
  pz=polarizability[aabbx6.iloc[(i,0)]], polarizability[aabbx6.iloc[(i,1)]], polarizability[aabbx6.iloc[(i,2)]], polarizability[aabbx6.iloc[(i,3)]], polarizability[aabbx6.iloc[(i,4)]]
  data12.append(pz)

PZ_aabbx6=(np.array(data12))

# Specific Heat

data13=[]
for i in range (0, abx3_datasize):
  sh=specific_heat[abx3.iloc[(i,0)]], specific_heat[abx3.iloc[(i,1)]], specific_heat[abx3.iloc[(i,2)]]
  data13.append(sh)

SH_abx3=(np.array(data13))

data13=[]
for i in range (0, a2bbx6_datasize):
  sh=specific_heat[a2bbx6.iloc[(i,0)]], specific_heat[a2bbx6.iloc[(i,1)]], specific_heat[a2bbx6.iloc[(i,2)]], specific_heat[a2bbx6.iloc[(i,3)]]
  data13.append(sh)

SH_a2bbx6=(np.array(data13))

data13=[]
for i in range (0, aabbx6_datasize):
  sh=specific_heat[aabbx6.iloc[(i,0)]], specific_heat[aabbx6.iloc[(i,1)]], specific_heat[aabbx6.iloc[(i,2)]], specific_heat[aabbx6.iloc[(i,3)]], specific_heat[aabbx6.iloc[(i,4)]]
  data13.append(sh)

SH_aabbx6=(np.array(data13))

# Thermal Conductivity

data14=[]
for i in range (0, abx3_datasize):
  tc=thermal_conductivity[abx3.iloc[(i,0)]], thermal_conductivity[abx3.iloc[(i,1)]], thermal_conductivity[abx3.iloc[(i,2)]]
  data14.append(tc)

TC_abx3=(np.array(data14))

data14=[]
for i in range (0, a2bbx6_datasize):
  tc=thermal_conductivity[a2bbx6.iloc[(i,0)]], thermal_conductivity[a2bbx6.iloc[(i,1)]], thermal_conductivity[a2bbx6.iloc[(i,2)]], thermal_conductivity[a2bbx6.iloc[(i,3)]]
  data14.append(tc)

TC_a2bbx6=(np.array(data14))

data14=[]
for i in range (0, aabbx6_datasize):
  tc=(thermal_conductivity[aabbx6.iloc[(i,0)]], thermal_conductivity[aabbx6.iloc[(i,1)]], thermal_conductivity[aabbx6.iloc[(i,2)]],
      thermal_conductivity[aabbx6.iloc[(i,3)]], thermal_conductivity[aabbx6.iloc[(i,4)]])
  data14.append(tc)

TC_aabbx6=(np.array(data14))

#ABX3

# Atomic Number
Z1 = []
for value in Z_abx3[:,0]:
	idx = [0 for _ in range(103)]
	idx[value] = 1
	Z1.append(idx)
Z1_abx3 = (np.array(Z1)).reshape(abx3_datasize,103,1)

Z2 = []
for value in Z_abx3[:,1]:
	idx = [0 for _ in range(103)]
	idx[value] = 1
	Z2.append(idx)
Z2_abx3 = np.array(Z2).reshape(abx3_datasize,103,1)

Z3 = []
for value in Z_abx3[:,2]:
	idx = [0 for _ in range(103)]
	idx[value] = 1
	Z3.append(idx)
Z3_abx3 = np.array(Z3).reshape(abx3_datasize,103,1)

zeropad  = np.zeros((4200,103,4))
Z_abx3 = np.concatenate((Z1_abx3, Z2_abx3, Z3_abx3, zeropad), -1)


# Group Number 
GN1 = []
for value in GN_abx3[:,0]:
	idx = [0 for _ in range(18)]
	idx[value] = 1
	GN1.append(idx)
GN1_abx3 = np.array(GN1).reshape(abx3_datasize,18,1)

GN2 = []
for value in GN_abx3[:,1]:
	idx = [0 for _ in range(18)]
	idx[value] = 1
	GN2.append(idx)
GN2_abx3 = np.array(GN2).reshape(abx3_datasize,18,1)

GN3 = []
for value in GN_abx3[:,2]:
	idx = [0 for _ in range(18)]
	idx[value] = 1
	GN3.append(idx)
GN3_abx3 = np.array(GN3).reshape(abx3_datasize,18,1)

zeropad  = np.zeros((4200,18,4))
GN_abx3 = np.concatenate((GN1_abx3, GN2_abx3, GN3_abx3, zeropad), -1)

# Row Number

RN1 = []
for value in RN_abx3[:,0]:
	idx = [0 for _ in range(9)]
	idx[value] = 1
	RN1.append(idx)
RN1_abx3 = np.array(RN1).reshape(abx3_datasize,9,1)

RN2 = []
for value in RN_abx3[:,1]:
	idx = [0 for _ in range(9)]
	idx[value] = 1
	RN2.append(idx)
RN2_abx3 = np.array(RN2).reshape(abx3_datasize,9,1)

RN3 = []
for value in RN_abx3[:,2]:
	idx = [0 for _ in range(9)]
	idx[value] = 1
	RN3.append(idx)
RN3_abx3 = np.array(RN3).reshape(abx3_datasize,9,1)

zeropad  = np.zeros((4200,9,4))
RN_abx3 = np.concatenate((RN1_abx3, RN2_abx3, RN3_abx3, zeropad), -1)

# Valence
VL1 = []
for value in VL_abx3[:,0]:
	idx = [0 for _ in range(9)]
	idx[value] = 1
	VL1.append(idx)
VL1_abx3 = np.array(VL1).reshape(abx3_datasize,9,1)

VL2 = []
for value in VL_abx3[:,1]:
	idx = [0 for _ in range(9)]
	idx[value] = 1
	VL2.append(idx)
VL2_abx3 = np.array(VL2).reshape(abx3_datasize,9,1)

VL3 = []
for value in VL_abx3[:,2]:
	idx = [0 for _ in range(9)]
	idx[value] = 1
	VL3.append(idx)
VL3_abx3 = np.array(VL3).reshape(abx3_datasize,9,1)

zeropad  = np.zeros((4200,9,4))
VL_abx3 = np.concatenate((VL1_abx3, VL2_abx3, VL3_abx3, zeropad), -1)

# Block
BK1 = []
for value in BK_abx3[:,0]:
	idx = [0 for _ in range(4)]
	idx[value] = 1
	BK1.append(idx)
BK1_abx3 = np.array(BK1).reshape(abx3_datasize,4,1)

BK2 = []
for value in BK_abx3[:,1]:
	idx = [0 for _ in range(4)]
	idx[value] = 1
	BK2.append(idx)
BK2_abx3 = np.array(BK2).reshape(abx3_datasize,4,1)

BK3 = []
for value in BK_abx3[:,2]:
	idx = [0 for _ in range(4)]
	idx[value] = 1
	BK3.append(idx)
BK3_abx3 = np.array(BK3).reshape(abx3_datasize,4,1)

zeropad  = np.zeros((4200,4,4))
BK_abx3 = np.concatenate((BK1_abx3, BK2_abx3, BK3_abx3, zeropad), -1)


# Electronegativity
X_=((X_abx3-0.7)/(3.98-0.7))*10 # max & min X are 3.98 & 0.7, respectively.
X_=(np.array(np.where(X_==10, 9, X_))).astype(int)

X1 = []
for value in X_[:,0]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	X1.append(idx)
X1_abx3 = np.array(X1).reshape(abx3_datasize,10,1)

X2 = []
for value in X_[:,1]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	X2.append(idx)
X2_abx3 = np.array(X2).reshape(abx3_datasize,10,1)

X3 = []
for value in X_[:,2]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	X3.append(idx)
X3_abx3 = np.array(X3).reshape(abx3_datasize,10,1)

zeropad  = np.zeros((4200,10,4))
X_abx3 = np.concatenate((X1_abx3, X2_abx3, X3_abx3, zeropad), -1)

# Covalent Radius
CR_=((CR_abx3-0.28)/(2.6-0.28))*10 # max & min CR are 2.6 & 0.28, respectively.
CR_=(np.array(np.where(CR_==10, 9, CR_))).astype(int)

CR1 = []
for value in CR_[:,0]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	CR1.append(idx)
CR1_abx3 = np.array(CR1).reshape(abx3_datasize,10,1)

CR2 = []
for value in CR_[:,1]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	CR2.append(idx)
CR2_abx3 = np.array(CR2).reshape(abx3_datasize,10,1)

CR3 = []
for value in CR_[:,2]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	CR3.append(idx)
CR3_abx3 = np.array(CR3).reshape(abx3_datasize,10,1)

zeropad  = np.zeros((4200,10,4))
CR_abx3 = np.concatenate((CR1_abx3, CR2_abx3, CR3_abx3, zeropad), -1)

# Electron Affinity

EA_=((EA_abx3-(-2.33))/(3.612724-(-2.33)))*10 # max & min EA are 3.612724 & -2.33, respectively.
EA_=(np.array(np.where(EA_==10, 9, EA_))).astype(int)

EA1 = []
for value in EA_[:,0]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	EA1.append(idx)
EA1_abx3 = np.array(EA1).reshape(abx3_datasize,10,1)

EA2 = []
for value in EA_[:,1]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	EA2.append(idx)
EA2_abx3 = np.array(EA2).reshape(abx3_datasize,10,1)

EA3 = []
for value in EA_[:,2]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	EA3.append(idx)
EA3_abx3 = np.array(EA3).reshape(abx3_datasize,10,1)

zeropad  = np.zeros((4200,10,4))
EA_abx3 = np.concatenate((EA1_abx3, EA2_abx3, EA3_abx3, zeropad), -1)

# Average Ionic Radius
IR_=((IR_abx3-0)/(1.94-0))*10 # max & min IR are 1.94 & 0, respectively.
IR_=(np.array(np.where(IR_==10, 9, IR_))).astype(int)

IR1 = []
for value in IR_[:,0]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	IR1.append(idx)
IR1_abx3 = np.array(IR1).reshape(abx3_datasize,10,1)

IR2 = []
for value in IR_[:,1]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	IR2.append(idx)
IR2_abx3 = np.array(IR2).reshape(abx3_datasize,10,1)

IR3 = []
for value in IR_[:,2]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	IR3.append(idx)
IR3_abx3 = np.array(IR3).reshape(abx3_datasize,10,1)

zeropad  = np.zeros((4200,10,4))
IR_abx3 = np.concatenate((IR1_abx3, IR2_abx3, IR3_abx3, zeropad), -1)

# Polarizability

PZ_=((PZ_abx3-0.204956)/(59.42-0.204956))*10 # max & min PZ are 59.42 & 0.204956, respectively.
PZ_=(np.array(np.where(PZ_==10, 9, PZ_))).astype(int)

PZ1 = []
for value in PZ_[:,0]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	PZ1.append(idx)
PZ1_abx3 = np.array(PZ1).reshape(abx3_datasize,10,1)

PZ2 = []
for value in PZ_[:,1]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	PZ2.append(idx)
PZ2_abx3 = np.array(PZ2).reshape(abx3_datasize,10,1)

PZ3 = []
for value in PZ_[:,2]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	PZ3.append(idx)
PZ3_abx3 = np.array(PZ3).reshape(abx3_datasize,10,1)

zeropad  = np.zeros((4200,10,4))
PZ_abx3 = np.concatenate((PZ1_abx3, PZ2_abx3, PZ3_abx3, zeropad), -1)

# Molar Volume
MV_=(8.2752*np.log10(MV_abx3)-5.3165).astype(int) # Coverted to Log scale. Max & min MV are 70.94 & 4.39, respectively.

MV1 = []
for value in MV_[:,0]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	MV1.append(idx)
MV1_abx3 = np.array(MV1).reshape(abx3_datasize,10,1)

MV2 = []
for value in MV_[:,1]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	MV2.append(idx)
MV2_abx3 = np.array(MV2).reshape(abx3_datasize,10,1)

MV3 = []
for value in MV_[:,2]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	MV3.append(idx)
MV3_abx3 = np.array(MV3).reshape(abx3_datasize,10,1)

zeropad  = np.zeros((4200,10,4))
MV_abx3 = np.concatenate((MV1_abx3, MV2_abx3, MV3_abx3, zeropad), -1)

# Ionization Energy
IE_=(12.49488*np.log10(IE_abx3)-7.37678).astype(int) # Coverted to Log scale. Max & min IE are 24.58741 & 3.8939, respectively.

IE1 = []
for value in IE_[:,0]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	IE1.append(idx)
IE1_abx3 = np.array(IE1).reshape(abx3_datasize,10,1)

IE2 = []
for value in IE_[:,1]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	IE2.append(idx)
IE2_abx3 = np.array(IE2).reshape(abx3_datasize,10,1)

IE3 = []
for value in IE_[:,2]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	IE3.append(idx)
IE3_abx3 = np.array(IE3).reshape(abx3_datasize,10,1)

zeropad  = np.zeros((4200,10,4))
IE_abx3 = np.concatenate((IE1_abx3, IE2_abx3, IE3_abx3, zeropad), -1)

# Thermal Conductivity

TC_=(1.97007*np.log10(TC_abx3)+4.81188).astype(int) # Coverted to Log scale. Max & min TC are 430 & 0.00361, respectively.

TC1 = []
for value in TC_[:,0]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	TC1.append(idx)
TC1_abx3 = np.array(TC1).reshape(abx3_datasize,10,1)

TC2 = []
for value in TC_[:,1]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	TC2.append(idx)
TC2_abx3 = np.array(TC2).reshape(abx3_datasize,10,1)

TC3 = []
for value in TC_[:,2]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	TC3.append(idx)
TC3_abx3 = np.array(TC3).reshape(abx3_datasize,10,1)

zeropad  = np.zeros((4200,10,4))
TC_abx3 = np.concatenate((TC1_abx3, TC2_abx3, TC3_abx3, zeropad), -1)

# Specific Heat
SH_=(4.24427*np.log10(SH_abx3)+5.0959).astype(int) # Coverted to Log scale. Max & min TC are 14.304 & 0.063, respectively.

SH1 = []
for value in SH_[:,0]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	SH1.append(idx)
SH1_abx3 = np.array(SH1).reshape(abx3_datasize,10,1)

SH2 = []
for value in SH_[:,1]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	SH2.append(idx)
SH2_abx3 = np.array(SH2).reshape(abx3_datasize,10,1)

SH3 = []
for value in SH_[:,2]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	SH3.append(idx)
SH3_abx3 = np.array(SH3).reshape(abx3_datasize,10,1)

zeropad  = np.zeros((4200,10,4))
SH_abx3 = np.concatenate((SH1_abx3, SH2_abx3, SH3_abx3, zeropad), -1)

#A2BBX6

# Atomic Number
Z1 = []
for value in Z_a2bbx6[:,0]:
	idx = [0 for _ in range(103)]
	idx[value] = 1
	Z1.append(idx)
Z1_a2bbx6 = (np.array(Z1)).reshape(a2bbx6_datasize,103,1)

Z2 = []
for value in Z_a2bbx6[:,1]:
	idx = [0 for _ in range(103)]
	idx[value] = 1
	Z2.append(idx)
Z2_a2bbx6 = np.array(Z2).reshape(a2bbx6_datasize,103,1)

Z3 = []
for value in Z_a2bbx6[:,2]:
	idx = [0 for _ in range(103)]
	idx[value] = 1
	Z3.append(idx)
Z3_a2bbx6 = np.array(Z3).reshape(a2bbx6_datasize,103,1)

Z4 = []
for value in Z_a2bbx6[:,3]:
	idx = [0 for _ in range(103)]
	idx[value] = 1
	Z4.append(idx)
Z4_a2bbx6 = np.array(Z4).reshape(a2bbx6_datasize,103,1)

zeropad  = np.zeros((a2bbx6_datasize,103,3))
Z_a2bbx6 = np.concatenate((Z1_a2bbx6, Z2_a2bbx6, Z3_a2bbx6, Z4_a2bbx6, zeropad), -1)


# Group Number 
GN1 = []
for value in GN_a2bbx6[:,0]:
	idx = [0 for _ in range(18)]
	idx[value] = 1
	GN1.append(idx)
GN1_a2bbx6 = np.array(GN1).reshape(a2bbx6_datasize,18,1)

GN2 = []
for value in GN_a2bbx6[:,1]:
	idx = [0 for _ in range(18)]
	idx[value] = 1
	GN2.append(idx)
GN2_a2bbx6 = np.array(GN2).reshape(a2bbx6_datasize,18,1)

GN3 = []
for value in GN_a2bbx6[:,2]:
	idx = [0 for _ in range(18)]
	idx[value] = 1
	GN3.append(idx)
GN3_a2bbx6 = np.array(GN3).reshape(a2bbx6_datasize,18,1)

GN4 = []
for value in GN_a2bbx6[:,3]:
	idx = [0 for _ in range(18)]
	idx[value] = 1
	GN4.append(idx)
GN4_a2bbx6 = np.array(GN4).reshape(a2bbx6_datasize,18,1)

zeropad  = np.zeros((a2bbx6_datasize,18,3))
GN_a2bbx6 = np.concatenate((GN1_a2bbx6, GN2_a2bbx6, GN3_a2bbx6, GN4_a2bbx6, zeropad), -1)


# Row Number
RN1 = []
for value in RN_a2bbx6[:,0]:
	idx = [0 for _ in range(9)]
	idx[value] = 1
	RN1.append(idx)
RN1_a2bbx6 = np.array(RN1).reshape(a2bbx6_datasize,9,1)

RN2 = []
for value in RN_a2bbx6[:,1]:
	idx = [0 for _ in range(9)]
	idx[value] = 1
	RN2.append(idx)
RN2_a2bbx6 = np.array(RN2).reshape(a2bbx6_datasize,9,1)

RN3 = []
for value in RN_a2bbx6[:,2]:
	idx = [0 for _ in range(9)]
	idx[value] = 1
	RN3.append(idx)
RN3_a2bbx6 = np.array(RN3).reshape(a2bbx6_datasize,9,1)

RN4 = []
for value in RN_a2bbx6[:,3]:
	idx = [0 for _ in range(9)]
	idx[value] = 1
	RN4.append(idx)
RN4_a2bbx6 = np.array(RN4).reshape(a2bbx6_datasize,9,1)

zeropad  = np.zeros((a2bbx6_datasize,9,3))
RN_a2bbx6 = np.concatenate((RN1_a2bbx6, RN2_a2bbx6, RN3_a2bbx6, RN4_a2bbx6, zeropad), -1)

# Valence
VL1 = []
for value in VL_a2bbx6[:,0]:
	idx = [0 for _ in range(9)]
	idx[value] = 1
	VL1.append(idx)
VL1_a2bbx6 = np.array(VL1).reshape(a2bbx6_datasize,9,1)

VL2 = []
for value in VL_a2bbx6[:,1]:
	idx = [0 for _ in range(9)]
	idx[value] = 1
	VL2.append(idx)
VL2_a2bbx6 = np.array(VL2).reshape(a2bbx6_datasize,9,1)

VL3 = []
for value in VL_a2bbx6[:,2]:
	idx = [0 for _ in range(9)]
	idx[value] = 1
	VL3.append(idx)
VL3_a2bbx6 = np.array(VL3).reshape(a2bbx6_datasize,9,1)

VL4 = []
for value in VL_a2bbx6[:,3]:
	idx = [0 for _ in range(9)]
	idx[value] = 1
	VL4.append(idx)
VL4_a2bbx6 = np.array(VL4).reshape(a2bbx6_datasize,9,1)

zeropad  = np.zeros((a2bbx6_datasize,9,3))
VL_a2bbx6 = np.concatenate((VL1_a2bbx6, VL2_a2bbx6, VL3_a2bbx6, VL4_a2bbx6, zeropad), -1)

# Block
BK1 = []
for value in BK_a2bbx6[:,0]:
	idx = [0 for _ in range(4)]
	idx[value] = 1
	BK1.append(idx)
BK1_a2bbx6 = np.array(BK1).reshape(a2bbx6_datasize,4,1)

BK2 = []
for value in BK_a2bbx6[:,1]:
	idx = [0 for _ in range(4)]
	idx[value] = 1
	BK2.append(idx)
BK2_a2bbx6 = np.array(BK2).reshape(a2bbx6_datasize,4,1)

BK3 = []
for value in BK_a2bbx6[:,2]:
	idx = [0 for _ in range(4)]
	idx[value] = 1
	BK3.append(idx)
BK3_a2bbx6 = np.array(BK3).reshape(a2bbx6_datasize,4,1)

BK4 = []
for value in BK_a2bbx6[:,3]:
	idx = [0 for _ in range(4)]
	idx[value] = 1
	BK4.append(idx)
BK4_a2bbx6 = np.array(BK4).reshape(a2bbx6_datasize,4,1)

zeropad  = np.zeros((a2bbx6_datasize,4,3))
BK_a2bbx6 = np.concatenate((BK1_a2bbx6, BK2_a2bbx6, BK3_a2bbx6, BK4_a2bbx6, zeropad), -1)


# Electronegativity
X_=((X_a2bbx6-0.7)/(3.98-0.7))*10 # max & min X are 3.98 & 0.7, respectively.
X_=(np.array(np.where(X_==10, 9, X_))).astype(int)

X1 = []
for value in X_[:,0]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	X1.append(idx)
X1_a2bbx6 = np.array(X1).reshape(a2bbx6_datasize,10,1)

X2 = []
for value in X_[:,1]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	X2.append(idx)
X2_a2bbx6 = np.array(X2).reshape(a2bbx6_datasize,10,1)

X3 = []
for value in X_[:,2]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	X3.append(idx)
X3_a2bbx6 = np.array(X3).reshape(a2bbx6_datasize,10,1)

X4 = []
for value in X_[:,3]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	X4.append(idx)
X4_a2bbx6 = np.array(X4).reshape(a2bbx6_datasize,10,1)

zeropad  = np.zeros((a2bbx6_datasize,10,3))
X_a2bbx6 = np.concatenate((X1_a2bbx6, X2_a2bbx6, X3_a2bbx6, X4_a2bbx6, zeropad), -1)

# Covalent Radius
CR_=((CR_a2bbx6-0.28)/(2.6-0.28))*10 # max & min CR are 2.6 & 0.28, respectively.
CR_=(np.array(np.where(CR_==10, 9, CR_))).astype(int)

CR1 = []
for value in CR_[:,0]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	CR1.append(idx)
CR1_a2bbx6 = np.array(CR1).reshape(a2bbx6_datasize,10,1)

CR2 = []
for value in CR_[:,1]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	CR2.append(idx)
CR2_a2bbx6 = np.array(CR2).reshape(a2bbx6_datasize,10,1)

CR3 = []
for value in CR_[:,2]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	CR3.append(idx)
CR3_a2bbx6 = np.array(CR3).reshape(a2bbx6_datasize,10,1)

CR4 = []
for value in CR_[:,3]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	CR4.append(idx)
CR4_a2bbx6 = np.array(CR4).reshape(a2bbx6_datasize,10,1)

zeropad  = np.zeros((a2bbx6_datasize,10,3))
CR_a2bbx6 = np.concatenate((CR1_a2bbx6, CR2_a2bbx6, CR3_a2bbx6, CR4_a2bbx6, zeropad), -1)

# Electron Affinity
EA_=((EA_a2bbx6-(-2.33))/(3.612724-(-2.33)))*10 # max & min EA are 3.612724 & -2.33, respectively.
EA_=(np.array(np.where(EA_==10, 9, EA_))).astype(int)

EA1 = []
for value in EA_[:,0]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	EA1.append(idx)
EA1_a2bbx6 = np.array(EA1).reshape(a2bbx6_datasize,10,1)

EA2 = []
for value in EA_[:,1]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	EA2.append(idx)
EA2_a2bbx6 = np.array(EA2).reshape(a2bbx6_datasize,10,1)

EA3 = []
for value in EA_[:,2]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	EA3.append(idx)
EA3_a2bbx6 = np.array(EA3).reshape(a2bbx6_datasize,10,1)

EA4 = []
for value in EA_[:,3]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	EA4.append(idx)
EA4_a2bbx6 = np.array(EA4).reshape(a2bbx6_datasize,10,1)

zeropad  = np.zeros((a2bbx6_datasize,10,3))
EA_a2bbx6 = np.concatenate((EA1_a2bbx6, EA2_a2bbx6, EA3_a2bbx6, EA4_a2bbx6, zeropad), -1)

# Average Ionic Radius
IR_=((IR_a2bbx6-0)/(1.94-0))*10 # max & min IR are 1.94 & 0, respectively.
IR_=(np.array(np.where(IR_==10, 9, IR_))).astype(int)

IR1 = []
for value in IR_[:,0]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	IR1.append(idx)
IR1_a2bbx6 = np.array(IR1).reshape(a2bbx6_datasize,10,1)

IR2 = []
for value in IR_[:,1]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	IR2.append(idx)
IR2_a2bbx6 = np.array(IR2).reshape(a2bbx6_datasize,10,1)

IR3 = []
for value in IR_[:,2]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	IR3.append(idx)
IR3_a2bbx6 = np.array(IR3).reshape(a2bbx6_datasize,10,1)

IR4 = []
for value in IR_[:,3]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	IR4.append(idx)
IR4_a2bbx6 = np.array(IR4).reshape(a2bbx6_datasize,10,1)

zeropad  = np.zeros((a2bbx6_datasize,10,3))
IR_a2bbx6 = np.concatenate((IR1_a2bbx6, IR2_a2bbx6, IR3_a2bbx6, IR4_a2bbx6, zeropad), -1)

# Polarizability
PZ_=((PZ_a2bbx6-0.204956)/(59.42-0.204956))*10 # max & min PZ are 59.42 & 0.204956, respectively.
PZ_=(np.array(np.where(PZ_==10, 9, PZ_))).astype(int)

PZ1 = []
for value in PZ_[:,0]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	PZ1.append(idx)
PZ1_a2bbx6 = np.array(PZ1).reshape(a2bbx6_datasize,10,1)

PZ2 = []
for value in PZ_[:,1]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	PZ2.append(idx)
PZ2_a2bbx6 = np.array(PZ2).reshape(a2bbx6_datasize,10,1)

PZ3 = []
for value in PZ_[:,2]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	PZ3.append(idx)
PZ3_a2bbx6 = np.array(PZ3).reshape(a2bbx6_datasize,10,1)

PZ4 = []
for value in PZ_[:,3]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	PZ4.append(idx)
PZ4_a2bbx6 = np.array(PZ4).reshape(a2bbx6_datasize,10,1)

zeropad  = np.zeros((a2bbx6_datasize,10,3))
PZ_a2bbx6 = np.concatenate((PZ1_a2bbx6, PZ2_a2bbx6, PZ3_a2bbx6, PZ4_a2bbx6, zeropad), -1)

# Molar Volume
MV_=(8.2752*np.log10(MV_a2bbx6)-5.3165).astype(int) # Coverted to Log scale. Max & min MV are 70.94 & 4.39, respectively.

MV1 = []
for value in MV_[:,0]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	MV1.append(idx)
MV1_a2bbx6 = np.array(MV1).reshape(a2bbx6_datasize,10,1)

MV2 = []
for value in MV_[:,1]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	MV2.append(idx)
MV2_a2bbx6 = np.array(MV2).reshape(a2bbx6_datasize,10,1)

MV3 = []
for value in MV_[:,2]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	MV3.append(idx)
MV3_a2bbx6 = np.array(MV3).reshape(a2bbx6_datasize,10,1)

MV4 = []
for value in MV_[:,3]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	MV4.append(idx)
MV4_a2bbx6 = np.array(MV4).reshape(a2bbx6_datasize,10,1)

zeropad  = np.zeros((a2bbx6_datasize,10,3))
MV_a2bbx6 = np.concatenate((MV1_a2bbx6, MV2_a2bbx6, MV3_a2bbx6, MV4_a2bbx6, zeropad), -1)

# Ionization Energy
IE_=(12.49488*np.log10(IE_a2bbx6)-7.37678).astype(int) # Coverted to Log scale. Max & min IE are 24.58741 & 3.8939, respectively.

IE1 = []
for value in IE_[:,0]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	IE1.append(idx)
IE1_a2bbx6 = np.array(IE1).reshape(a2bbx6_datasize,10,1)

IE2 = []
for value in IE_[:,1]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	IE2.append(idx)
IE2_a2bbx6 = np.array(IE2).reshape(a2bbx6_datasize,10,1)

IE3 = []
for value in IE_[:,2]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	IE3.append(idx)
IE3_a2bbx6 = np.array(IE3).reshape(a2bbx6_datasize,10,1)

IE4 = []
for value in IE_[:,3]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	IE4.append(idx)
IE4_a2bbx6 = np.array(IE4).reshape(a2bbx6_datasize,10,1)

zeropad  = np.zeros((a2bbx6_datasize,10,3))
IE_a2bbx6 = np.concatenate((IE1_a2bbx6, IE2_a2bbx6, IE3_a2bbx6, IE4_a2bbx6, zeropad), -1)

# Thermal Conductivity

TC_=(1.97007*np.log10(TC_a2bbx6)+4.81188).astype(int) # Coverted to Log scale. Max & min TC are 430 & 0.00361, respectively.

TC1 = []
for value in TC_[:,0]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	TC1.append(idx)
TC1_a2bbx6 = np.array(TC1).reshape(a2bbx6_datasize,10,1)

TC2 = []
for value in TC_[:,1]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	TC2.append(idx)
TC2_a2bbx6 = np.array(TC2).reshape(a2bbx6_datasize,10,1)

TC3 = []
for value in TC_[:,2]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	TC3.append(idx)
TC3_a2bbx6 = np.array(TC3).reshape(a2bbx6_datasize,10,1)

TC4 = []
for value in TC_[:,3]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	TC4.append(idx)
TC4_a2bbx6 = np.array(TC4).reshape(a2bbx6_datasize,10,1)

zeropad  = np.zeros((a2bbx6_datasize,10,3))
TC_a2bbx6 = np.concatenate((TC1_a2bbx6, TC2_a2bbx6, TC3_a2bbx6, TC4_a2bbx6, zeropad), -1)

# Specific Heat
SH_=(4.24427*np.log10(SH_a2bbx6)+5.0959).astype(int) # Coverted to Log scale. Max & min TC are 14.304 & 0.063, respectively.

SH1 = []
for value in SH_[:,0]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	SH1.append(idx)
SH1_a2bbx6 = np.array(SH1).reshape(a2bbx6_datasize,10,1)

SH2 = []
for value in SH_[:,1]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	SH2.append(idx)
SH2_a2bbx6 = np.array(SH2).reshape(a2bbx6_datasize,10,1)

SH3 = []
for value in SH_[:,2]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	SH3.append(idx)
SH3_a2bbx6 = np.array(SH3).reshape(a2bbx6_datasize,10,1)

SH4 = []
for value in SH_[:,3]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	SH4.append(idx)
SH4_a2bbx6 = np.array(SH4).reshape(a2bbx6_datasize,10,1)

zeropad  = np.zeros((a2bbx6_datasize,10,3))
SH_a2bbx6 = np.concatenate((SH1_a2bbx6, SH2_a2bbx6, SH3_a2bbx6, SH4_a2bbx6, zeropad), -1)

#AABBX6

# Atomic Number
Z1 = []
for value in Z_aabbx6[:,0]:
	idx = [0 for _ in range(103)]
	idx[value] = 1
	Z1.append(idx)
Z1_aabbx6 = (np.array(Z1)).reshape(aabbx6_datasize,103,1)

Z2 = []
for value in Z_aabbx6[:,1]:
	idx = [0 for _ in range(103)]
	idx[value] = 1
	Z2.append(idx)
Z2_aabbx6 = np.array(Z2).reshape(aabbx6_datasize,103,1)

Z3 = []
for value in Z_aabbx6[:,2]:
	idx = [0 for _ in range(103)]
	idx[value] = 1
	Z3.append(idx)
Z3_aabbx6 = np.array(Z3).reshape(aabbx6_datasize,103,1)

Z4 = []
for value in Z_aabbx6[:,3]:
	idx = [0 for _ in range(103)]
	idx[value] = 1
	Z4.append(idx)
Z4_aabbx6 = np.array(Z4).reshape(aabbx6_datasize,103,1)

Z5 = []
for value in Z_aabbx6[:,4]:
	idx = [0 for _ in range(103)]
	idx[value] = 1
	Z5.append(idx)
Z5_aabbx6 = np.array(Z5).reshape(aabbx6_datasize,103,1)

zeropad  = np.zeros((aabbx6_datasize,103,2))
Z_aabbx6 = np.concatenate((Z1_aabbx6, Z2_aabbx6, Z3_aabbx6, Z4_aabbx6, Z5_aabbx6, zeropad), -1)


# Group Number 
GN1 = []
for value in GN_aabbx6[:,0]:
	idx = [0 for _ in range(18)]
	idx[value] = 1
	GN1.append(idx)
GN1_aabbx6 = np.array(GN1).reshape(aabbx6_datasize,18,1)

GN2 = []
for value in GN_aabbx6[:,1]:
	idx = [0 for _ in range(18)]
	idx[value] = 1
	GN2.append(idx)
GN2_aabbx6 = np.array(GN2).reshape(aabbx6_datasize,18,1)

GN3 = []
for value in GN_aabbx6[:,2]:
	idx = [0 for _ in range(18)]
	idx[value] = 1
	GN3.append(idx)
GN3_aabbx6 = np.array(GN3).reshape(aabbx6_datasize,18,1)

GN4 = []
for value in GN_aabbx6[:,3]:
	idx = [0 for _ in range(18)]
	idx[value] = 1
	GN4.append(idx)
GN4_aabbx6 = np.array(GN4).reshape(aabbx6_datasize,18,1)

GN5 = []
for value in GN_aabbx6[:,4]:
	idx = [0 for _ in range(18)]
	idx[value] = 1
	GN5.append(idx)
GN5_aabbx6 = np.array(GN5).reshape(aabbx6_datasize,18,1)

zeropad  = np.zeros((aabbx6_datasize,18,2))
GN_aabbx6 = np.concatenate((GN1_aabbx6, GN2_aabbx6, GN3_aabbx6, GN4_aabbx6, GN5_aabbx6, zeropad), -1)


# Row Number
RN1 = []
for value in RN_aabbx6[:,0]:
	idx = [0 for _ in range(9)]
	idx[value] = 1
	RN1.append(idx)
RN1_aabbx6 = np.array(RN1).reshape(aabbx6_datasize,9,1)

RN2 = []
for value in RN_aabbx6[:,1]:
	idx = [0 for _ in range(9)]
	idx[value] = 1
	RN2.append(idx)
RN2_aabbx6 = np.array(RN2).reshape(aabbx6_datasize,9,1)

RN3 = []
for value in RN_aabbx6[:,2]:
	idx = [0 for _ in range(9)]
	idx[value] = 1
	RN3.append(idx)
RN3_aabbx6 = np.array(RN3).reshape(aabbx6_datasize,9,1)

RN4 = []
for value in RN_aabbx6[:,3]:
	idx = [0 for _ in range(9)]
	idx[value] = 1
	RN4.append(idx)
RN4_aabbx6 = np.array(RN4).reshape(aabbx6_datasize,9,1)

RN5 = []
for value in RN_aabbx6[:,4]:
	idx = [0 for _ in range(9)]
	idx[value] = 1
	RN5.append(idx)
RN5_aabbx6 = np.array(RN5).reshape(aabbx6_datasize,9,1)

zeropad  = np.zeros((aabbx6_datasize,9,2))
RN_aabbx6 = np.concatenate((RN1_aabbx6, RN2_aabbx6, RN3_aabbx6, RN4_aabbx6, RN5_aabbx6, zeropad), -1)

# Valence
VL1 = []
for value in VL_aabbx6[:,0]:
	idx = [0 for _ in range(9)]
	idx[value] = 1
	VL1.append(idx)
VL1_aabbx6 = np.array(VL1).reshape(aabbx6_datasize,9,1)

VL2 = []
for value in VL_aabbx6[:,1]:
	idx = [0 for _ in range(9)]
	idx[value] = 1
	VL2.append(idx)
VL2_aabbx6 = np.array(VL2).reshape(aabbx6_datasize,9,1)

VL3 = []
for value in VL_aabbx6[:,2]:
	idx = [0 for _ in range(9)]
	idx[value] = 1
	VL3.append(idx)
VL3_aabbx6 = np.array(VL3).reshape(aabbx6_datasize,9,1)

VL4 = []
for value in VL_aabbx6[:,3]:
	idx = [0 for _ in range(9)]
	idx[value] = 1
	VL4.append(idx)
VL4_aabbx6 = np.array(VL4).reshape(aabbx6_datasize,9,1)

VL5 = []
for value in VL_aabbx6[:,4]:
	idx = [0 for _ in range(9)]
	idx[value] = 1
	VL5.append(idx)
VL5_aabbx6 = np.array(VL5).reshape(aabbx6_datasize,9,1)

zeropad  = np.zeros((aabbx6_datasize,9,2))
VL_aabbx6 = np.concatenate((VL1_aabbx6, VL2_aabbx6, VL3_aabbx6, VL4_aabbx6, VL5_aabbx6, zeropad), -1)

# Block
BK1 = []
for value in BK_aabbx6[:,0]:
	idx = [0 for _ in range(4)]
	idx[value] = 1
	BK1.append(idx)
BK1_aabbx6 = np.array(BK1).reshape(aabbx6_datasize,4,1)

BK2 = []
for value in BK_aabbx6[:,1]:
	idx = [0 for _ in range(4)]
	idx[value] = 1
	BK2.append(idx)
BK2_aabbx6 = np.array(BK2).reshape(aabbx6_datasize,4,1)

BK3 = []
for value in BK_aabbx6[:,2]:
	idx = [0 for _ in range(4)]
	idx[value] = 1
	BK3.append(idx)
BK3_aabbx6 = np.array(BK3).reshape(aabbx6_datasize,4,1)

BK4 = []
for value in BK_aabbx6[:,3]:
	idx = [0 for _ in range(4)]
	idx[value] = 1
	BK4.append(idx)
BK4_aabbx6 = np.array(BK4).reshape(aabbx6_datasize,4,1)

BK5 = []
for value in BK_aabbx6[:,4]:
	idx = [0 for _ in range(4)]
	idx[value] = 1
	BK5.append(idx)
BK5_aabbx6 = np.array(BK5).reshape(aabbx6_datasize,4,1)

zeropad  = np.zeros((aabbx6_datasize,4,2))
BK_aabbx6 = np.concatenate((BK1_aabbx6, BK2_aabbx6, BK3_aabbx6, BK4_aabbx6, BK5_aabbx6, zeropad), -1)


# Electronegativity
X_=((X_aabbx6-0.7)/(3.98-0.7))*10 # max & min X are 3.98 & 0.7, respectively.
X_=(np.array(np.where(X_==10, 9, X_))).astype(int)

X1 = []
for value in X_[:,0]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	X1.append(idx)
X1_aabbx6 = np.array(X1).reshape(aabbx6_datasize,10,1)

X2 = []
for value in X_[:,1]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	X2.append(idx)
X2_aabbx6 = np.array(X2).reshape(aabbx6_datasize,10,1)

X3 = []
for value in X_[:,2]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	X3.append(idx)
X3_aabbx6 = np.array(X3).reshape(aabbx6_datasize,10,1)

X4 = []
for value in X_[:,3]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	X4.append(idx)
X4_aabbx6 = np.array(X4).reshape(aabbx6_datasize,10,1)

X5 = []
for value in X_[:,4]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	X5.append(idx)
X5_aabbx6 = np.array(X5).reshape(aabbx6_datasize,10,1)

zeropad  = np.zeros((aabbx6_datasize,10,2))
X_aabbx6 = np.concatenate((X1_aabbx6, X2_aabbx6, X3_aabbx6, X4_aabbx6, X5_aabbx6, zeropad), -1)

# Covalent Radius
CR_=((CR_aabbx6-0.28)/(2.6-0.28))*10 # max & min CR are 2.6 & 0.28, respectively.
CR_=(np.array(np.where(CR_==10, 9, CR_))).astype(int)

CR1 = []
for value in CR_[:,0]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	CR1.append(idx)
CR1_aabbx6 = np.array(CR1).reshape(aabbx6_datasize,10,1)

CR2 = []
for value in CR_[:,1]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	CR2.append(idx)
CR2_aabbx6 = np.array(CR2).reshape(aabbx6_datasize,10,1)

CR3 = []
for value in CR_[:,2]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	CR3.append(idx)
CR3_aabbx6 = np.array(CR3).reshape(aabbx6_datasize,10,1)

CR4 = []
for value in CR_[:,3]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	CR4.append(idx)
CR4_aabbx6 = np.array(CR4).reshape(aabbx6_datasize,10,1)

CR5 = []
for value in CR_[:,4]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	CR5.append(idx)
CR5_aabbx6 = np.array(CR5).reshape(aabbx6_datasize,10,1)

zeropad  = np.zeros((aabbx6_datasize,10,2))
CR_aabbx6 = np.concatenate((CR1_aabbx6, CR2_aabbx6, CR3_aabbx6, CR4_aabbx6, CR5_aabbx6, zeropad), -1)

# Electron Affinity
EA_=((EA_aabbx6-(-2.33))/(3.612724-(-2.33)))*10 # max & min EA are 3.612724 & -2.33, respectively.
EA_=(np.array(np.where(EA_==10, 9, EA_))).astype(int)

EA1 = []
for value in EA_[:,0]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	EA1.append(idx)
EA1_aabbx6 = np.array(EA1).reshape(aabbx6_datasize,10,1)

EA2 = []
for value in EA_[:,1]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	EA2.append(idx)
EA2_aabbx6 = np.array(EA2).reshape(aabbx6_datasize,10,1)

EA3 = []
for value in EA_[:,2]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	EA3.append(idx)
EA3_aabbx6 = np.array(EA3).reshape(aabbx6_datasize,10,1)

EA4 = []
for value in EA_[:,3]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	EA4.append(idx)
EA4_aabbx6 = np.array(EA4).reshape(aabbx6_datasize,10,1)

EA5 = []
for value in EA_[:,4]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	EA5.append(idx)
EA5_aabbx6 = np.array(EA5).reshape(aabbx6_datasize,10,1)

zeropad  = np.zeros((aabbx6_datasize,10,2))
EA_aabbx6 = np.concatenate((EA1_aabbx6, EA2_aabbx6, EA3_aabbx6, EA4_aabbx6, EA5_aabbx6, zeropad), -1)

# Average Ionic Radius
IR_=((IR_aabbx6-0)/(1.94-0))*10 # max & min IR are 1.94 & 0, respectively.
IR_=(np.array(np.where(IR_==10, 9, IR_))).astype(int)

IR1 = []
for value in IR_[:,0]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	IR1.append(idx)
IR1_aabbx6 = np.array(IR1).reshape(aabbx6_datasize,10,1)

IR2 = []
for value in IR_[:,1]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	IR2.append(idx)
IR2_aabbx6 = np.array(IR2).reshape(aabbx6_datasize,10,1)

IR3 = []
for value in IR_[:,2]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	IR3.append(idx)
IR3_aabbx6 = np.array(IR3).reshape(aabbx6_datasize,10,1)

IR4 = []
for value in IR_[:,3]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	IR4.append(idx)
IR4_aabbx6 = np.array(IR3).reshape(aabbx6_datasize,10,1)

IR5 = []
for value in IR_[:,4]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	IR5.append(idx)
IR5_aabbx6 = np.array(IR5).reshape(aabbx6_datasize,10,1)

zeropad  = np.zeros((aabbx6_datasize,10,2))
IR_aabbx6 = np.concatenate((IR1_aabbx6, IR2_aabbx6, IR3_aabbx6, IR4_aabbx6, IR5_aabbx6, zeropad), -1)

# Polarizability
PZ_=((PZ_aabbx6-0.204956)/(59.42-0.204956))*10 # max & min PZ are 59.42 & 0.204956, respectively.
PZ_=(np.array(np.where(PZ_==10, 9, PZ_))).astype(int)

PZ1 = []
for value in PZ_[:,0]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	PZ1.append(idx)
PZ1_aabbx6 = np.array(PZ1).reshape(aabbx6_datasize,10,1)

PZ2 = []
for value in PZ_[:,1]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	PZ2.append(idx)
PZ2_aabbx6 = np.array(PZ2).reshape(aabbx6_datasize,10,1)

PZ3 = []
for value in PZ_[:,2]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	PZ3.append(idx)
PZ3_aabbx6 = np.array(PZ3).reshape(aabbx6_datasize,10,1)

PZ4 = []
for value in PZ_[:,3]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	PZ4.append(idx)
PZ4_aabbx6 = np.array(PZ4).reshape(aabbx6_datasize,10,1)

PZ5 = []
for value in PZ_[:,4]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	PZ5.append(idx)
PZ5_aabbx6 = np.array(PZ5).reshape(aabbx6_datasize,10,1)

zeropad  = np.zeros((aabbx6_datasize,10,2))
PZ_aabbx6 = np.concatenate((PZ1_aabbx6, PZ2_aabbx6, PZ3_aabbx6, PZ4_aabbx6, PZ5_aabbx6, zeropad), -1)

# Molar Volume
MV_=(8.2752*np.log10(MV_aabbx6)-5.3165).astype(int) # Coverted to Log scale. Max & min MV are 70.94 & 4.39, respectively.

MV1 = []
for value in MV_[:,0]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	MV1.append(idx)
MV1_aabbx6 = np.array(MV1).reshape(aabbx6_datasize,10,1)

MV2 = []
for value in MV_[:,1]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	MV2.append(idx)
MV2_aabbx6 = np.array(MV2).reshape(aabbx6_datasize,10,1)

MV3 = []
for value in MV_[:,2]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	MV3.append(idx)
MV3_aabbx6 = np.array(MV3).reshape(aabbx6_datasize,10,1)

MV4 = []
for value in MV_[:,3]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	MV4.append(idx)
MV4_aabbx6 = np.array(MV4).reshape(aabbx6_datasize,10,1)

MV5 = []
for value in MV_[:,4]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	MV5.append(idx)
MV5_aabbx6 = np.array(MV5).reshape(aabbx6_datasize,10,1)

zeropad  = np.zeros((aabbx6_datasize,10,2))
MV_aabbx6 = np.concatenate((MV1_aabbx6, MV2_aabbx6, MV3_aabbx6, MV4_aabbx6, MV5_aabbx6, zeropad), -1)

# Ionization Energy
IE_=(12.49488*np.log10(IE_aabbx6)-7.37678).astype(int) # Coverted to Log scale. Max & min IE are 24.58741 & 3.8939, respectively.

IE1 = []
for value in IE_[:,0]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	IE1.append(idx)
IE1_aabbx6 = np.array(IE1).reshape(aabbx6_datasize,10,1)

IE2 = []
for value in IE_[:,1]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	IE2.append(idx)
IE2_aabbx6 = np.array(IE2).reshape(aabbx6_datasize,10,1)

IE3 = []
for value in IE_[:,2]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	IE3.append(idx)
IE3_aabbx6 = np.array(IE3).reshape(aabbx6_datasize,10,1)

IE4 = []
for value in IE_[:,3]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	IE4.append(idx)
IE4_aabbx6 = np.array(IE4).reshape(aabbx6_datasize,10,1)

IE5 = []
for value in IE_[:,4]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	IE5.append(idx)
IE5_aabbx6 = np.array(IE5).reshape(aabbx6_datasize,10,1)

zeropad  = np.zeros((aabbx6_datasize,10,2))
IE_aabbx6 = np.concatenate((IE1_aabbx6, IE2_aabbx6, IE3_aabbx6, IE4_aabbx6, IE5_aabbx6, zeropad), -1)

# Thermal Conductivity

TC_=(1.97007*np.log10(TC_aabbx6)+4.81188).astype(int) # Coverted to Log scale. Max & min TC are 430 & 0.00361, respectively.

TC1 = []
for value in TC_[:,0]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	TC1.append(idx)
TC1_aabbx6 = np.array(TC1).reshape(aabbx6_datasize,10,1)

TC2 = []
for value in TC_[:,1]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	TC2.append(idx)
TC2_aabbx6 = np.array(TC2).reshape(aabbx6_datasize,10,1)

TC3 = []
for value in TC_[:,2]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	TC3.append(idx)
TC3_aabbx6 = np.array(TC3).reshape(aabbx6_datasize,10,1)

TC4 = []
for value in TC_[:,3]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	TC4.append(idx)
TC4_aabbx6 = np.array(TC4).reshape(aabbx6_datasize,10,1)

TC5 = []
for value in TC_[:,4]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	TC5.append(idx)
TC5_aabbx6 = np.array(TC5).reshape(aabbx6_datasize,10,1)

zeropad  = np.zeros((aabbx6_datasize,10,2))
TC_aabbx6 = np.concatenate((TC1_aabbx6, TC2_aabbx6, TC3_aabbx6, TC4_aabbx6, TC5_aabbx6, zeropad), -1)

# Specific Heat
SH_=(4.24427*np.log10(SH_aabbx6)+5.0959).astype(int) # Coverted to Log scale. Max & min TC are 14.304 & 0.063, respectively.

SH1 = []
for value in SH_[:,0]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	SH1.append(idx)
SH1_aabbx6 = np.array(SH1).reshape(aabbx6_datasize,10,1)

SH2 = []
for value in SH_[:,1]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	SH2.append(idx)
SH2_aabbx6 = np.array(SH2).reshape(aabbx6_datasize,10,1)

SH3 = []
for value in SH_[:,2]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	SH3.append(idx)
SH3_aabbx6 = np.array(SH3).reshape(aabbx6_datasize,10,1)

SH4 = []
for value in SH_[:,3]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	SH4.append(idx)
SH4_aabbx6 = np.array(SH4).reshape(aabbx6_datasize,10,1)

SH5 = []
for value in SH_[:,4]:
	idx = [0 for _ in range(10)]
	idx[value] = 1
	SH5.append(idx)
SH5_aabbx6 = np.array(SH5).reshape(aabbx6_datasize,10,1)

zeropad  = np.zeros((aabbx6_datasize,10,2))
SH_aabbx6 = np.concatenate((SH1_aabbx6, SH2_aabbx6, SH3_aabbx6, SH4_aabbx6, SH5_aabbx6, zeropad), -1)

Z_array = np.row_stack((Z_abx3,Z_a2bbx6,Z_aabbx6))
GN_array = np.row_stack((GN_abx3,GN_a2bbx6,GN_aabbx6))
RN_array = np.row_stack((RN_abx3,RN_a2bbx6,RN_aabbx6))
X_array = np.row_stack((X_abx3,X_a2bbx6,X_aabbx6))
CR_array = np.row_stack((CR_abx3,CR_a2bbx6,CR_aabbx6))
VL_array = np.row_stack((VL_abx3,VL_a2bbx6,VL_aabbx6))
IE_array = np.row_stack((IE_abx3,IE_a2bbx6,IE_aabbx6))
EA_array = np.row_stack((EA_abx3,EA_a2bbx6,EA_aabbx6))
BK_array = np.row_stack((BK_abx3,BK_a2bbx6,BK_aabbx6))
MV_array = np.row_stack((MV_abx3,MV_a2bbx6,MV_aabbx6))
IR_array = np.row_stack((IR_abx3,IR_a2bbx6,IR_aabbx6))
PZ_array = np.row_stack((PZ_abx3,PZ_a2bbx6,PZ_aabbx6))
SH_array = np.row_stack((SH_abx3,SH_a2bbx6,SH_aabbx6))
TC_array = np.row_stack((TC_abx3,TC_a2bbx6,TC_aabbx6))

PROP_array = np.hstack((GN_array,RN_array,X_array,CR_array,VL_array,IE_array,EA_array,BK_array,MV_array,IR_array,PZ_array,SH_array,TC_array))
PROP_array.shape


SO = np.array(pd.read_csv('site_occupancy.csv'))

data=[]
for i in range(SO.shape[0]):
    SO1=SO[i,0:5]
    SO2=SO[i,5:10]
    SO3=SO[i,10:15]
    SO4=SO[i,15:20]
    SO5=SO[i,20:25]
    SO_=np.column_stack((SO1,SO2,SO3,SO4,SO5))
    data.append(SO_)
SO=np.array(data)

zeropad = np.zeros((SO.shape[0],5,2))
SO = np.concatenate((SO,zeropad), -1)
SO.shape

STOIC = (np.asarray(pd.read_csv('stoic.csv').astype('float32'))).reshape(PROP_array.shape[0],1,3)
zeropad = np.zeros((STOIC.shape[0],1,4))
STOIC = np.concatenate((STOIC,zeropad), -1)

NA = (np.asarray(pd.read_csv('n_atoms.csv').astype('float32'))).reshape(PROP_array.shape[0],1,7)

from sklearn.preprocessing import MinMaxScaler

LV = np.asarray(pd.read_csv('lattice_params.csv').astype('float32'))
scaler_LV = MinMaxScaler()
scaler_LV.fit(LV)
LV=(scaler_LV.transform(LV)).reshape(PROP_array.shape[0],1,3)
zeropad = np.zeros((LV.shape[0],1,4))
LV = np.concatenate((LV,zeropad), -1)

ANG = np.asarray(pd.read_csv('angle.csv').astype('float32'))
scaler_ANG = MinMaxScaler()
scaler_ANG.fit(ANG)
ANG=(scaler_ANG.transform(ANG)).reshape(PROP_array.shape[0],1,3)
zeropad = np.zeros((ANG.shape[0],1,4))
ANG = np.concatenate((ANG,zeropad), -1)

AC = np.asarray(pd.read_csv('atom_coord.csv').astype('float32'))
scaler_AC = MinMaxScaler()
scaler_AC.fit(AC)
AC=scaler_AC.transform(AC)

data=[]
for i in range(AC.shape[0]):
    AC1=AC[i,0:3]
    AC2=AC[i,3:6]
    AC3=AC[i,6:9]
    AC4=AC[i,9:12]
    AC5=AC[i,12:15]
    AC6=AC[i,15:18]
    AC7=AC[i,18:21]
    AC8=AC[i,21:24]
    AC9=AC[i,24:27]
    AC10=AC[i,27:30]
    AC11=AC[i,30:33]
    AC12=AC[i,33:36]
    AC13=AC[i,36:39]
    AC14=AC[i,39:42]
    AC15=AC[i,42:45]
    AC16=AC[i,45:48]
    AC17=AC[i,48:51]
    AC18=AC[i,51:54]
    AC19=AC[i,54:57]
    AC20=AC[i,57:60]
    AC21=AC[i,60:63]
    AC22=AC[i,63:66]
    AC23=AC[i,66:69]
    AC24=AC[i,69:72]
    AC25=AC[i,72:75]
    AC26=AC[i,75:78]
    AC27=AC[i,78:81]
    AC28=AC[i,81:84]
    AC29=AC[i,84:87]
    AC30=AC[i,87:90]
    AC31=AC[i,90:93]
    AC32=AC[i,93:96]
    AC33=AC[i,96:99]
    AC34=AC[i,99:102]
    AC35=AC[i,102:105]
    AC36=AC[i,105:108]
    AC37=AC[i,108:111]
    AC38=AC[i,111:114]
    AC39=AC[i,114:117]
    AC40=AC[i,117:120]

    AC_=np.row_stack((AC1,AC2,AC3,AC4,AC5,AC6,AC7,AC8,AC9,AC10,AC11,AC12,AC13,AC14,AC15,AC16,AC17,AC18,AC19,AC20,
                      AC21,AC22,AC23,AC24,AC25,AC26,AC27,AC28,AC29,AC30,AC31,AC32,AC33,AC34,AC35,AC36,AC37,AC38,AC39,AC40))
    data.append(AC_)
AC=np.array(data)

zeropad = np.zeros((AC.shape[0],40,4))
AC = np.concatenate((AC, zeropad), -1)

zeropad = np.zeros((PROP_array.shape[0],1,7))
input_imgs = np.hstack((Z_array, STOIC, SO, LV, ANG, NA, AC, PROP_array))
zeropad = np.zeros((PROP_array.shape[0],282,1))
input_imgs = np.concatenate((input_imgs, zeropad), -1)

# RGB Image representation
X_samples = input_imgs.reshape(PROP_array.shape[0],94,8,3)

# Processing Target variables
df = pd.read_csv('target.csv')
y = np.asarray(df.drop(["pretty_formula","spacegroup.symbol","crystal_system"], 1)).astype('float32')

# ICSD label
y_icsd = y[:,0]

# Formation energy
y_Ef = y[:,4]
y_Ef_class = y[:,5]

# Energy above convex hull
y_Ehull = y[:,6]
y_Ehull_class = y[:,7]

# Stoichiometry
y_stoic = y[:,13]

# Number of atoms in unit cell
y_atoms = y[:,21]
