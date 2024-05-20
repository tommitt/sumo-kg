import pickle

from sumo.agent import LlmAgent
from sumo.schemas import Ontology

# inputs
ontology_labels = [
    {"Persona": "Persone fisiche"},
    {"Azienda": "Soggetti aziendali e organizzazioni"},
    "Documento",
    "Attività",
    "Software",
    "Altro",
]
ontology_relationships = ["Relation between any pair of entities"]

full_text = """\
Medicolavoro.org è una piattaforma digitale che mette a rete i medici del lavoro ed eroga servizi di medicina del lavoro nei confronti delle aziende clienti.
Ora ti descrivo il processo all'interno di Medico Lavoro.
In una prima fase, il cliente raggiunge medico lavoro.org tramite email; la richiesta viene qualificata da Stefano.
Se supera questa fase, si capisce se il cliente è servibile, molto spesso a seconda della zona geografica.
Successivamente, ci si chiede se si tratti realmente di un’azienda o di un’agenzia di servizi.
Se è un'agenzia, si invia anche un listino prezzi oltre al preventivo;
se invece è un'azienda, si invia solo un preventivo e, se accetta, diventa un cliente.
A questo punto, il processo successivo all'acquisizione del cliente da parte di Medico Lavoro.org si articola in tre diverse sottofasi:
una prima fase di pre-analisi, una fase di analisi e una fase esecutiva.
La prima fase di pre-analisi è gestita interamente da Stefano e inizia con l'accettazione del preventivo;
dopodiché Stefano si fa la seguente domanda: la città del cliente è coperta dai nostri medici?
Qualora la risposta fosse negativa, Stefano si troverebbe costretto a rifiutare il preventivo (si pensi al caso di un cliente che apre una nuova filiale in una città non coperta da Medico Lavoro.org).
Quando invece la città è coperta, Stefano si chiede se l'azienda cliente abbia più o meno di 15 dipendenti:
quando ne ha di più, c'è bisogno di organizzare la riunione periodica annuale;
quando ne ha di meno, non è necessaria alcuna azione aggiuntiva.
La fase successiva è quella di analisi ed è gestita dall'ufficio operativo composto da Christian e Andrea.
Questa sottofase inizia con Christian che si occupa della raccolta dei dati;
in particolare, i dati da raccogliere sono il protocollo sanitario, il documento sulla valutazione dei rischi, la visura camerale e l'elenco dei lavoratori.
Non tutti questi documenti sono obbligatori; possiamo suddividere i documenti in bloccanti e non bloccanti.
I documenti non bloccanti, in particolare, sono il protocollo sanitario - perché questo può anche dover essere redatto da zero - e il DVR.
Infatti, il DVR, nonostante sia obbligatorio per il datore di lavoro, non è responsabilità del medico: il medico ha la responsabilità di non permettere danni alla salute ai dipendenti, non ha responsabilità sul DVR se non segnalare la questione al datore di lavoro.
Una volta superata la fase di raccolta, analisi e valutazione dei dati, si passa alla fase di nomina o co-nomina a seconda del caso.
La co-nomina avviene quando l'azienda ha già un medico nella sede principale, ma per una nuova sede in una nuova città c'è quindi bisogno di nominare un nuovo medico in co-nomina col medico principale e questa casistica è meno costosa della nomina.
La nomina invece comporta la stesura del protocollo sanitario e la verifica del preventivo effettuato in precedenza.
In questa fase si verificano le mansioni, si legge il DVR, si intervista il datore di lavoro… si verifica insomma che i dati raccolti siano coerenti con il preventivo.
Si redige così il protocollo sanitario, steso dallo staff principalmente da Christian con il supporto eventuale di Stefano, Sveto e solo in ultimo il medico.
Una volta redatto il protocollo sanitario, quest'ultimo si inserisce nel software gestionale chiamato Achille.
Una volta inserito il protocollo sanitario nel software gestionale, Andrea si occupa dell'impostazione delle visite e delle trasferte.
L'impostazione delle visite delle trasferte è basata sul protocollo sanitario quindi sui dati dei dipendenti e sulla loro mansione.
Le visite possono essere con prelievo o senza.
Quando sono con prelievo, il prelievo viene effettuato sempre a domicilio, quindi viene assegnato un infermiere che si occupa del prelievo e che infine emette i referti.
Quando invece le visite sono senza prelievo, possono essere a domicilio o in sede. La scelta è ad esclusiva preferenza del datore di lavoro.
Quando le visite senza prelievo si effettuano a domicilio, bisogna chiedersi: è la prima volta che se ne effettua una a domicilio?
Se è la prima volta, è necessario anche un sopralluogo; se non è la prima volta, si passa direttamente all’assegnazione del dottore.
Ovviamente, quando le visite senza prelievo sono svolte in sede, si passa direttamente all'assegnazione del dottore.
Dal punto di vista legale, la situazione è un po' tirata; infatti, la legge prevede che il medico competente sia anche quello assegnato.
Tuttavia, Medico Lavoro.org ipotizza che tutti i propri medici siano bravi uguali e quindi non necessariamente il medico che ha ricevuto la nomina sarà anche quello effettivamente assegnato.
Inoltre, talvolta il cliente richiede un medico specifico. È già in ogni caso presente un sistema di assegnazione dei medici costruito sulla base dei feedback dei clienti.
Quando in ultimo il dottore è assegnato, egli svolge le visite.
Le visite, in combinazione con i referti effettuati dagli infermieri, sono necessari per la preparazione delle idoneità.\
"""

# create kg
ontology = Ontology(labels=ontology_labels, relationships=ontology_relationships)
agent = LlmAgent(ontology=ontology)
agent_state = agent.run(full_text)

# save graph
out_filename = ".scratchpad/outputs/graph.pkl"
with open(out_filename, "wb") as f:
    pickle.dump(agent_state["kg"], f)
