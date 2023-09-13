from pickle import FALSE, TRUE
from queue import Empty
from experta import *
from Utils import *
from MaternalHealthRisk import *
from dataAnalysis import *
import array as arr

#strutture complicazioni
complicazioni_lista = []
complicazioni_sintomi = []
fattore_rischio = {}
complicazioni_desc = {}
complicazioni_tratt = {}

def caricamento():
    global complicazioni_lista,complicazioni_sintomi,fattore_rischio,complicazioni_desc,complicazioni_tratt
    complicazioni = open("complicazioni.txt")
    complicazioni_t = complicazioni.read()
    complicazioni_lista = complicazioni_t.split("\n")
    complicazioni.close()
    for complicazione in complicazioni_lista:
        complicazione_s_file = open("Complicazioni rischi/" + complicazione + ".txt")
        complicazione_s_data = complicazione_s_file.read()
        s_lista = complicazione_s_data.split("\n")
        complicazioni_sintomi.append(s_lista)
        fattore_rischio[str(s_lista)] = complicazione
        complicazione_s_file.close()
        complicazione_s_file = open("Complicazioni descrizione/" + complicazione + ".txt")
        complicazione_s_data = complicazione_s_file.read()
        complicazioni_desc[complicazione] = complicazione_s_data
        complicazione_s_file.close()
        complicazione_s_file = open("Complicazioni trattamento/" + complicazione + ".txt")
        complicazione_s_data = complicazione_s_file.read()
        complicazioni_tratt[complicazione] = complicazione_s_data
        complicazione_s_file.close()

def get_descrizione(complicazione):
    return complicazioni_desc[complicazione]

def get_trattamento(complicazione):
    return complicazioni_tratt[complicazione]

def if_not_matched(complicazione):
        if complicazione!="":
            print("")
            id_complicazione = complicazione
            complicazione_descrizione = get_descrizione(id_complicazione)
            complicazione_trattamento = get_trattamento(id_complicazione)
            print("")
            print("___________________________")
            print("\nNessuna complicazione combacia con i tuoi esatti fattori di rischio.")
            print("La complicazione che rischi di piu' e' %s\n" %(id_complicazione))
            print("Una breve descrizione della complicazione:\n")
            print(complicazione_descrizione+"\n")
            print("Il trattamento consigliato secondo la NIH: \n")
            print(complicazione_trattamento+"\n")
            print("___________________________")
        else:
            print("\nNon hai nessun sintomo o fattore di rischio quindi non rischi nessuna complicazione.\n")

# Regole esperte
class SistemaEsperto(KnowledgeEngine):
    @DefFacts()
    def _initial_action(self):
        print("\nTi verranno posti alcuni fattori di rischio o sintomi, rispondi 'si' nel caso interessi te o la tua esperienza, 'no' altrimenti.\n")
        yield Fact(action="trova_complicazioni")

    @Rule(Fact(action='trova_complicazioni'), NOT(Fact(obesita=W())),salience = 1)
    def fattore_rischio_0(self):
        risposta=" "
        while(risposta != "si" and risposta != "no"):
            risposta=input("Obesita': ").lower()
            if risposta != "si" and risposta != "no":
                print("\nrispondi solo con si o no!\n") 
        self.declare(Fact(obesita=risposta))

    @Rule(Fact(action='trova_complicazioni'), NOT(Fact(contrazioni_addominali=W())),salience = 1)
    def fattore_rischio_1(self):
        risposta=" "
        while(risposta != "si" and risposta != "no"):
            risposta=input("Contrazioni addominali: ").lower()
            if risposta != "si" and risposta != "no":
                print("\nrispondi solo con si o no!\n") 
        self.declare(Fact(contrazioni_addominali=risposta))

    @Rule(Fact(action='trova_complicazioni'), NOT(Fact(bruciore_stomaco=W())),salience = 1)
    def fattore_rischio_2(self):
        risposta=" "
        while(risposta != "si" and risposta != "no"):
            risposta=input("Bruciore di stomaco: ").lower()
            if risposta != "si" and risposta != "no":
                print("\nrispondi solo con si o no!\n") 
        self.declare(Fact(bruciore_stomaco=risposta))

    @Rule(Fact(action='trova_complicazioni'), NOT(Fact(pressione_alta=W())),salience = 1)
    def fattore_rischio_3(self):
        risposta=" "
        while(risposta != "si" and risposta != "no"):
            risposta=input("Aumento della pressione: ").lower()
            if risposta != "si" and risposta != "no":
                print("\nrispondi solo con si o no!\n") 
        self.declare(Fact(pressione_alta=risposta))

    @Rule(Fact(action='trova_complicazioni'), NOT(Fact(vista_sfocata=W())),salience = 1)
    def fattore_rischio_4(self):
        risposta=" "
        while(risposta != "si" and risposta != "no"):
            risposta=input("Vista sfocata: ").lower()
            if risposta != "si" and risposta != "no":
                print("\nrispondi solo con si o no!\n") 
        self.declare(Fact(vista_sfocata=risposta))

    @Rule(Fact(action='trova_complicazioni'), NOT(Fact(nausea=W())),salience = 1)
    def fattore_rischio_5(self):
        risposta=" "
        while(risposta != "si" and risposta != "no"):
            risposta=input("Nausea: ").lower()
            if risposta != "si" and risposta != "no":
                print("\nrispondi solo con si o no!\n") 
        self.declare(Fact(nausea=risposta))
    
    @Rule(Fact(action='trova_complicazioni'), NOT(Fact(infezioni=W())),salience = 1)
    def fattore_rischio_6(self):
        risposta=" "
        while(risposta != "si" and risposta != "no"):
            risposta=input("Infezioni: ").lower()
            if risposta != "si" and risposta != "no":
                print("\nrispondi solo con si o no!\n") 
        self.declare(Fact(infezioni=risposta))
    
    @Rule(Fact(action='trova_complicazioni'), NOT(Fact(perdita_sangue=W())),salience = 1)
    def fattore_rischio_7(self):
        risposta=" "
        while(risposta != "si" and risposta != "no"):
            risposta=input("Perdita di sangue: ").lower()
            if risposta != "si" and risposta != "no":
                print("\nrispondi solo con si o no!\n") 
        self.declare(Fact(perdita_sangue=risposta))
    
    @Rule(Fact(action='trova_complicazioni'), NOT(Fact(perdita_liquidi=W())),salience = 1)
    def fattore_rischio_8(self):
        risposta=" "
        while(risposta != "si" and risposta != "no"):
            risposta=input("Perdita di liquidi: ").lower()
            if risposta != "si" and risposta != "no":
                print("\nrispondi solo con si o no!\n") 
        self.declare(Fact(perdita_liquidi=risposta))
    
    @Rule(Fact(action='trova_complicazioni'), NOT(Fact(vomito=W())),salience = 1)
    def fattore_rischio_9(self):
        risposta=" "
        while(risposta != "si" and risposta != "no"):
            risposta=input("Vomito: ").lower()
            if risposta != "si" and risposta != "no":
                print("\nrispondi solo con si o no!\n") 
        self.declare(Fact(vomito=risposta))

    @Rule(Fact(action='trova_complicazioni'), NOT(Fact(perdita_peso_appetito=W())),salience = 1)
    def fattore_rischio_10(self):
        risposta=" "
        while(risposta != "si" and risposta != "no"):
            risposta=input("Perdita di peso e/o appetito: ").lower()
            if risposta != "si" and risposta != "no":
                print("\nrispondi solo con si o no!\n") 
        self.declare(Fact(perdita_peso_appetito=risposta))

    @Rule(Fact(action='trova_complicazioni'), NOT(Fact(sensazione_svenimento=W())),salience = 1)
    def fattore_rischio_11(self):
        risposta=" "
        while(risposta != "si" and risposta != "no"):
            risposta=input("Sensazione di svenimento: ").lower()
            if risposta != "si" and risposta != "no":
                print("\nrispondi solo con si o no!\n") 
        self.declare(Fact(sensazione_svenimento=risposta))

    @Rule(Fact(action='trova_complicazioni'), NOT(Fact(respiro_corto=W())),salience = 1)
    def fattore_rischio_12(self):
        risposta=" "
        while(risposta != "si" and risposta != "no"):
            risposta=input("Respiro corto: ").lower()
            if risposta != "si" and risposta != "no":
                print("\nrispondi solo con si o no!\n") 
        self.declare(Fact(respiro_corto=risposta))

    @Rule(Fact(action='trova_complicazioni'), NOT(Fact(stanchezza=W())),salience = 1)
    def fattore_rischio_13(self):
        risposta=" "
        while(risposta != "si" and risposta != "no"):
            risposta=input("Stanchezza: ").lower()
            if risposta != "si" and risposta != "no":
                print("\nrispondi solo con si o no!\n") 
        self.declare(Fact(stanchezza=risposta))

    @Rule(Fact(action='trova_complicazioni'),Fact(obesita="no"),Fact(contrazioni_addominali="no"),Fact(bruciore_stomaco="no"),Fact(pressione_alta="no"),Fact(vista_sfocata="no"),Fact(nausea="no"),Fact(infezioni="no"),Fact(perdita_sangue="no"),Fact(perdita_liquidi="no"),Fact(vomito="no"),Fact(perdita_peso_appetito="no"),Fact(sensazione_svenimento="si"),Fact(respiro_corto="si"),Fact(stanchezza="si"),salience = 1)
    def complicazione_0(self):
        self.declare(Fact(complicazione="anemia ferro"))

    @Rule(Fact(action='trova_complicazioni'),Fact(obesita="si"),Fact(contrazioni_addominali="no"),Fact(bruciore_stomaco="no"),Fact(pressione_alta="no"),Fact(vista_sfocata="no"),Fact(nausea="no"),Fact(infezioni="no"),Fact(perdita_sangue="no"),Fact(perdita_liquidi="no"),Fact(vomito="no"),Fact(perdita_peso_appetito="no"),Fact(sensazione_svenimento="no"),Fact(respiro_corto="no"),Fact(stanchezza="no"),salience = 2)
    def complicazione_1(self):
        self.declare(Fact(complicazione="diabete gestazionale"))

    @Rule(Fact(action='trova_complicazioni'),Fact(obesita="no"),Fact(contrazioni_addominali="no"),Fact(bruciore_stomaco="no"),Fact(pressione_alta="no"),Fact(vista_sfocata="no"),Fact(nausea="si"),Fact(infezioni="no"),Fact(perdita_sangue="no"),Fact(perdita_liquidi="no"),Fact(vomito="si"),Fact(perdita_peso_appetito="si"),Fact(sensazione_svenimento="si"),Fact(respiro_corto="no"),Fact(stanchezza="no"),salience = 2)
    def complicazione_2(self):
        self.declare(Fact(complicazione="iperemesi gravidica"))

    @Rule(Fact(action='trova_complicazioni'),Fact(obesita="no"),Fact(contrazioni_addominali="si"),Fact(bruciore_stomaco="no"),Fact(pressione_alta="no"),Fact(vista_sfocata="no"),Fact(nausea="no"),Fact(infezioni="no"),Fact(perdita_sangue="si"),Fact(perdita_liquidi="si"),Fact(vomito="no"),Fact(perdita_peso_appetito="no"),Fact(sensazione_svenimento="no"),Fact(respiro_corto="no"),Fact(stanchezza="no"),salience = 3)
    def complicazione_3(self):
        self.declare(Fact(complicazione="mortinatalita"))

    @Rule(Fact(action='trova_complicazioni'),Fact(obesita="no"),Fact(contrazioni_addominali="si"),Fact(bruciore_stomaco="no"),Fact(pressione_alta="no"),Fact(vista_sfocata="no"),Fact(nausea="si"),Fact(infezioni="si"),Fact(perdita_sangue="si"),Fact(perdita_liquidi="si"),Fact(vomito="no"),Fact(perdita_peso_appetito="no"),Fact(sensazione_svenimento="no"),Fact(respiro_corto="no"),Fact(stanchezza="no"),salience = 2)
    def complicazione_4(self):
        self.declare(Fact(complicazione="travaglio pretermine"))

    @Rule(Fact(action='trova_complicazioni'),Fact(obesita="si"),Fact(contrazioni_addominali="si"),Fact(bruciore_stomaco="si"),Fact(pressione_alta="si"),Fact(vista_sfocata="si"),Fact(nausea="si"),Fact(infezioni="no"),Fact(perdita_sangue="no"),Fact(perdita_liquidi="no"),Fact(vomito="no"),Fact(perdita_peso_appetito="no"),Fact(sensazione_svenimento="no"),Fact(respiro_corto="no"),Fact(stanchezza="no"),salience = 1)
    def complicazione_5(self):
        self.declare(Fact(complicazione="preeclampsia"))

    

    @Rule(Fact(action='trova_complicazioni'),Fact(complicazione=MATCH.complicazione),salience =1)
    def complicazione(self, complicazione):
        print("")
        id_complicazione = complicazione
        complicazione_descrizione = get_descrizione(id_complicazione)
        complicazione_trattamento = get_trattamento(id_complicazione)
        print("")
        print("___________________________")
        print("La complicazione che rischi di piu' e' %s\n" %(id_complicazione))
        print("Una breve descrizione della complicazione:\n")
        print(complicazione_descrizione+"\n")
        print("Il trattamento consigliato secondo la NIH: \n")
        print(complicazione_trattamento+"\n")
        print("___________________________")

    @Rule(Fact(action='trova_complicazioni'),
          Fact(obesita=MATCH.obesita),
          Fact(contrazioni_addominali=MATCH.contrazioni_addominali),
          Fact(bruciore_stomaco=MATCH.bruciore_stomaco),
          Fact(pressione_alta=MATCH.pressione_alta),
          Fact(vista_sfocata=MATCH.vista_sfocata),
          Fact(nausea=MATCH.nausea),
          Fact(infezioni=MATCH.infezioni),
          Fact(perdita_sangue=MATCH.perdita_sangue),
          Fact(perdita_liquidi=MATCH.perdita_liquidi),
          Fact(vomito=MATCH.vomito),
          Fact(perdita_peso_appetito=MATCH.perdita_peso_appetito),
          Fact(sensazione_svenimento=MATCH.sensazione_svenimento),
          Fact(respiro_corto=MATCH.respiro_corto),
          Fact(stanchezza=MATCH.stanchezza),NOT(Fact(complicazione=MATCH.complicazione)),salience = -1)

    def not_matched(self, obesita, contrazioni_addominali, bruciore_stomaco, pressione_alta, vista_sfocata, nausea, infezioni, perdita_sangue, perdita_liquidi, vomito, perdita_peso_appetito, sensazione_svenimento, respiro_corto, stanchezza):
        lis = [obesita, contrazioni_addominali, bruciore_stomaco, pressione_alta, vista_sfocata, nausea, infezioni, perdita_sangue, perdita_liquidi, vomito, perdita_peso_appetito, sensazione_svenimento, respiro_corto, stanchezza]
        max = 0
        max_complicazione = ""
        for key,val in fattore_rischio.items():
            count = 0
            temp_list = eval(key)
            for j in range(0,len(lis)):
                if(temp_list[j] == lis[j] and lis[j] == "si"):
                    count = count + 1
            if count > max:
                max = count
                max_complicazione = val
        if_not_matched(max_complicazione)




if __name__ == "__main__":
    caricamento()
    sistema = SistemaEsperto()
    scelta = 1
    svcTrained = NULL
    rfTrained = NULL
    dtTrained = NULL
    knnTrained = NULL
    nbTrained = NULL
    while (scelta != 0):
        print("Cosa vuoi fare? \n 1. Analizza i tuoi valori 2. Inserisci i tuoi sintomi 0. Esci")
        scelta = int(input())
        if scelta==1:
            print("Vuoi vedere anche tutti i grafici sui dati registrati? (si/no)")
            dataShowRisp = ""
            while(dataShowRisp != "si" and dataShowRisp != "no"):
                dataShowRisp = (input(": ")).lower()
                if dataShowRisp != "si" and dataShowRisp != "no":
                    print("\nrispondi solo con si o no!\n") 

            print("Inserisci l'eta")
            eta = 0
            while(eta < 8 or eta > 80):
                eta = float(input(": ")) 
                if eta < 8 or eta > 80:
                    print("\nEta' consentita fra 8 e 80 anni\n") 
            print("Inserisci la pressione sistolica")
            sbp = 0
            while(sbp < 80 or sbp > 200):
                sbp = float(input(": ")) 
                if sbp < 80 or sbp > 200:
                    print("\nPressione sistolica consentita fra 80 e 200\n") 
            print("Inserisci la pressione distolica")
            dbp = 0
            while(dbp < 30 or dbp >= sbp):
                dbp = float(input(": ")) 
                if dbp < 30 or dbp >= sbp:
                    print("\nPressione distolica consentita fra 30 e ") 
                    print(sbp-1)
                    print("\n")
            print("Inserisci la glicemia")
            bs = 0
            while(bs < 3 or bs > 20):
                bs = float(input(": ")) 
                if bs < 3 or bs > 20:
                    print("\nGlicemia consentita fra 3 e 20\n") 
            print("Inserisci la temperatura corporea(Fahrenheit)")
            temp = float(input(": "))
            while(temp < 93 or temp > 106): 
                temp = float(input(": ")) 
                if temp < 93 or temp > 106:
                    print("\Temperature consentita fra 93(32 gradi) e 106(42 gradi)\n") 
            
            numbers_list = [eta, sbp, dbp, bs, temp]
            b = arr.array('d', [eta, sbp, dbp, bs, temp])
            a = np.array(b)
            sample = a.reshape(1, -1)

            if(dataShowRisp == "si"):
                dataShow()
            if(svcTrained == NULL or rfTrained == NULL or dtTrained == NULL or knnTrained == NULL or nbTrained == NULL):
                svcTrained,rfTrained,dtTrained,knnTrained,nbTrained = trainModels(dataShowRisp)
            
            predSvc = svcTrained.predict(sample)
            predRf = rfTrained.predict(sample)
            predDt = dtTrained.predict(sample)
            predKnn = knnTrained.predict(sample)
            predNb = nbTrained.predict(sample)
            print("Il rischio predetto dal classificatore con piu' accuratezza dai tuoi dati inseriti è:(0: basso, 1: medio, 2: alto)")
            print(predRf)
        if scelta==2:
            sistema.reset()
            sistema.run()
