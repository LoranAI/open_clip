# CLIPEX

## Introduzione
Qui scriverò ciò che ho fatto fino ad ora per tenere traccia del lavoro.

## Quello che ho pensato
Ho provato a far funzionare la prima implementazione di PolytopeTransformer sia per la parte VISION che quella TEXT, sembra funzionare tutto, da testare solo alla fine del training.
Quello che non mi torna è il layer intermediaro che ovviamente dovrà essere "Finetuning" anch'esso dato che si tratta di un nuovo layer con nuovi pesi, a differenza del fixed classifer simplex.

Poi ho pensato di fare, come detto anche da parte del biondi in un vecchio messaggio, di togliere il layer di proiezione già integrato nei VISIONTransformer e TEXTTransformer, cioè text_projection e proj e di aggiungere la nuova classe, che in questo caso ho riscritto come SIMPLEXCUSTOM, che non è altro che il layer che si era implementato ma senza controlli sul tipo di out_features, in modo da fare prima e che si adatti in base al tipo di politpo che scegli.
In questo modo forse si evita di portarci a presso tutte le classi di ereditarietà che abbiamo fatto, poi si valuta insieme, anche perchè con la vecchia implementazione avevamo "due proiezioni" cioè sia quella dei vari VISION E TEXT Transformer più la nostra del simplex.

IN NOTA: le modifiche che ho apportato sono: Quando modifico il codice di open_clip ho aggiunto la nota: ""# NOTA: CLIPEX"", in modo che se devi navigare nel codice trovi facilmente le cose modifiche. Alcune volte ho creato dei file simple_"nome_file" per separare quello che avevamo fatto noi da quello che c'era già, in modo da non creare confusione (Ma forse ho fatto comunque confusione).

Ho creato due args, uno per wandb entity, e una per l'uso di polytope, ma non ho capito perchè me lo prende sempre come vero, forse ho sbagliato qualcosa (Bestemmia).

Poi per il resto mi sembra banale l'esecuzione e vedere che cosa succede, dato che non stiamo modificando la dimensione di uscita (in entrambe le implementazioni) tutte le valutazioni dovrebbero essere corrette e il modello dovrebbe funzionare.

Ho anche aggiornato open_clip fork e mi sembra meglio, anche perchè toglie alcune dipendenze dal progetto e sminchiava sempre tutti i moduli.

## Per creare l'ambiente
Per creare l'ambiente, in modo tale che si possa cambiare anche macchina in caso di mancanza di gpu, ho creato un nuovo comando makefile che installa automaticamente tutto, ovviamente manca la chiave per wandb (l'organizzazione è già stata messa all'interno del wandb init, ma si può cambiare da arg)
Prima però usa questo:
```bash
python3 -m venv .env
source .env/bin/activate
pip install -U pip
```

poi fai:
```bash
make install-all
```

E metti la chiave di wandb se te la chiede, sennò bone. Così non ci dovrebbero essere più problemi di alcuna sorta, al massimo lo vedi te.
Ovviamente sono un ragazzo speciale e mi faceva mettere direttamente i pacchetti sul requirements.txt, ma non avevo sbatti e quindi ho messo direttamente il comando :D.

## Problemi
Ovviamente la prima evaluation delle due implementazioni non porta per ora a nulla dato che manca il finetuning, maledette gpu del cazzo che non ci sono mai.

## Considerazioni:
Io direi di chiarire come vogliamo fare, anche per come vogliamo proseguire, strutturare e rendere anche il codice più leggibile, se toglierci dal fork di open_clip e creare da zero una nuova repository per fare le cose come si deve, in modo da modificarlo come ci pare e piace, togliendo anche i vari commenti e file nuovi che ho creato per non creare confusione.

Io direi di togliere quasi tutto il codice di aro_dataset, tranne le valutazioni opportune, anche per togliere le dipendenze, ma vediamo più avanti.

Sistemare un po' il makefile se ce ne fosse bisogno.
Sistemare i yaml e i vari file di configurazione.
Usare gli args per fare le cose in modo più pulito.

Fare un test su clipex, o anche detto polytope clip, io direi di tenere anche CLIPEX come nome che al biondi piaceva.

Ah se vedi ho modificato il nome dell'organizzazione in LoranAI, dato che non mi piaceva il nome di prima, se non ti piace possiamo cambiarlo, ma mi sembrava più carino.

## COME E' composto il CLIP:
    CLIP:
        VisionTransformer:
            - ln_pre
            - transformer
            - ln_post
            - proj -> che io ho tolto in favore di SIMPLEXCUSTOM

        TextTransformer:
            - token_emb
            - pos_emb
            - transformer
            - ln_final
            - text_projection -> che io ho tolto in favore di SIMPLEXCUSTOM

Invece di:

    PolytopeCLIP:
        PolytopeVisionTransformer:
            - ln_pre
            - transformer
            - ln_post
            - proj
            - polytope -> che è il nostro layer SIMPLEX
        
        PolytopeTextTransformer:
            - token_emb
            - pos_emb
            - transformer
            - ln_final
            - text_projection
            - polytope -> che è il nostro layer SIMPLEX
