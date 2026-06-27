import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
import torch.nn.functional as F


class MatrixAutoencoder(nn.Module):
    """
    Autoencoder per dati tabulari (Probe Requests preprocessate).

    STRUTTURA:
      - Encoder: comprime il vettore di 140 feature in uno spazio latente
                 di dimensione emb_size (default 64). Più profondo rispetto
                 alla versione originale per catturare correlazioni non-lineari
                 tra gli Information Elements (IE).
      - Decoder: ricostruisce il vettore originale dallo spazio latente.
                 Usato solo durante il training per la reconstruction loss.

    TRAINING (fit_clustering):
      Si ottimizza una loss combinata:
        total_loss = surrogate_loss * (1 - recon_weight)
                   + recon_loss    *  recon_weight

      - surrogate_loss: guida la separazione tra device diversi nello
        spazio latente usando le true label degli scenari di training.
      - recon_loss: forza l'encoder a preservare le informazioni strutturali
        degli IE. Senza di essa, l'encoder potrebbe collassare tutto in un
        punto e soddisfare la surrogate loss in modo banale, perdendo la
        capacità di generalizzare su device mai visti in test.

    INFERENZA (encode_dataloader):
      Si passano le PR del test set attraverso l'encoder e si ottengono
      gli embedding. Non si applica nessuno scaling sugli embedding:
      le distanze euclidee nello spazio latente vengono usate direttamente
      da DBSCAN, il cui eps viene calibrato automaticamente con il metodo
      delle k-distanze (vedi estimate_eps).
    """

    def __init__(self, n_features, emb_size=64, hidden_dim=128):
        super().__init__()

        self.n_features = n_features
        self.emb_size   = emb_size
        self.hidden_dim = hidden_dim

        # -------------------------------------------------------------------
        # ENCODER: n_features → 256 → hidden_dim → emb_size
        #
        # BatchNorm1d dopo il primo layer: normalizza le attivazioni per
        # stabilizzare il training (utile perché le feature del preprocessing
        # hanno scale diverse: la maggior parte binarie 0/1, ma alcune
        # intere o float con range più ampi).
        #
        # Dropout(0.2): spegne casualmente il 20% dei neuroni ad ogni
        # forward pass durante il training, riducendo l'overfitting sugli
        # scenari di training.
        # -------------------------------------------------------------------
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_size)
        )

        # -------------------------------------------------------------------
        # DECODER: emb_size → hidden_dim → 256 → n_features
        # Speculare all'encoder. Usato solo per la reconstruction loss.
        # -------------------------------------------------------------------
        self.decoder = nn.Sequential(
            nn.Linear(emb_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n_features)
        )

    def forward(self, x):
        """
        Forward pass completo: input → encoder → decoder → output.
        Restituisce sia la ricostruzione x_hat che l'embedding z.
        """
        z     = self.encoder(x)      # (batch, emb_size)
        x_hat = self.decoder(z)      # (batch, n_features)
        return x_hat, z

    # -----------------------------------------------------------------------
    # METODI DI ENCODING
    # -----------------------------------------------------------------------

    def encode(self, x):
        """Codifica un singolo tensore x nello spazio latente (no grad)."""
        self.eval()
        with torch.no_grad():
            z = self.encoder(x.float())
        return z

    def encode_dataloader(self, dataloader, device=None):
        """
        Estrae gli embedding di tutti i campioni in un dataloader.

        Itera sul dataloader batch per batch, passa ogni batch attraverso
        l'encoder e accumula gli embedding in una lista, che viene poi
        concatenata in un unico tensore.

        Restituisce:
          - (embeddings, labels) se il dataloader fornisce le label
          - embeddings            altrimenti
        """
        if device is None:
            device = next(self.parameters()).device

        self.eval()
        embeddings = []
        labels     = []

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                    if len(batch) > 1:
                        labels.append(batch[1].cpu())
                else:
                    x = batch

                x = x.to(device).float()
                z = self.encoder(x)
                embeddings.append(z.cpu())

        embeddings = torch.cat(embeddings, dim=0)

        if len(labels) > 0:
            labels = torch.cat(labels, dim=0)
            return embeddings, labels

        return embeddings

    # -----------------------------------------------------------------------
    # STIMA AUTOMATICA DI EPS PER DBSCAN
    # -----------------------------------------------------------------------

    @staticmethod
    def estimate_eps(embeddings_np, k=4, percentile=90):
        """
        Stima automatica di eps per DBSCAN tramite il metodo delle k-distanze.

        IDEA:
          DBSCAN considera un punto "core" se ha almeno min_samples vicini
          entro raggio eps. Per scegliere eps in modo data-driven, si calcola
          per ogni punto la distanza al suo k-esimo vicino più prossimo
          (con k = min_samples). Ordinando queste distanze in modo crescente
          si ottiene la "k-distance curve": il punto di massima curvatura
          (il "gomito") indica il valore naturale di eps oltre il quale
          i punti cominciano ad essere considerati rumore.

          Anziché trovare il gomito geometricamente, usiamo il percentile
          della distribuzione delle k-distanze: questo dà un eps robusto
          che include la maggior parte dei punti nel clustering.

        Args:
            embeddings_np : array numpy (n_samples, emb_size)
            k             : numero di vicini = min_samples di DBSCAN
            percentile    : percentile della distribuzione k-distanze
                            (90 → include ~90% dei punti come core points)

        Returns:
            eps_estimated : float, valore di eps stimato
            k_distances   : array delle k-distanze ordinate (per eventuale
                            visualizzazione del gomito)
        """
        # NearestNeighbors trova i k vicini più prossimi per ogni punto
        # usando distanza euclidea nello spazio latente (senza scaling).
        nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(embeddings_np)

        # distances ha shape (n_samples, k): distanza di ogni punto
        # ai suoi k vicini. La colonna [:, k-1] è la distanza al k-esimo
        # vicino (il più lontano tra i k), che è quella rilevante per DBSCAN.
        distances, _ = nbrs.kneighbors(embeddings_np)
        k_distances  = np.sort(distances[:, k - 1])

        # Il percentile scelto della distribuzione è il nostro eps stimato.
        eps_estimated = float(np.percentile(k_distances, percentile))

        return eps_estimated, k_distances

    # -----------------------------------------------------------------------
    # TRAINING CON LOSS COMBINATA
    # -----------------------------------------------------------------------

    def fit_clustering(self, dataloader, epochs=10, lr=1e-3, device=None,
                       min_samples=5, recon_weight=0.3,
                       surrogate='prototype', temperature=0.5):
        """
        Addestra l'autoencoder ottimizzando clustering e ricostruzione insieme.

        FLUSSO PER OGNI EPOCH:
          1. Calcola gli embedding attuali (no grad) per valutare DBSCAN.
          2. Stima eps automaticamente con le k-distanze sugli embedding.
          3. Esegue DBSCAN e calcola l'ARI rispetto alle true label
             (solo per monitoraggio — non è differenziabile).
          4. Ricalcola gli embedding CON grad e calcola la surrogate loss
             (differenziabile) che guida la separazione tra device.
          5. Calcola la reconstruction loss per preservare le informazioni IE.
          6. Ottimizza la somma pesata delle due loss.

        Args:
            dataloader  : DataLoader che restituisce (x, y) con y = true label
            epochs      : numero di epoch di training
            lr          : learning rate per Adam
            min_samples : minimo campioni per formare un cluster in DBSCAN
                          (usato anche come k per la stima di eps)
            recon_weight: peso della reconstruction loss [0..1].
                          Consigliato: 0.3. Non azzerare: senza recon loss
                          l'encoder perde la capacità di generalizzare su
                          device mai visti negli scenari di test.
            surrogate   : tipo di loss di clustering differenziabile:
                          'prototype'  → cross-entropy rispetto ai prototipi
                                         di classe (media degli embedding)
                          'contrastive'→ NT-Xent loss
                          'fisher'     → rapporto scatter within/between class
            temperature : temperatura per le loss prototype e contrastive
                          (valori bassi = logits più netti = loss più dura)
        """
        if device is None:
            device = next(self.parameters()).device
        self.to(device)

        optimizer      = torch.optim.Adam(self.parameters(), lr=lr)
        recon_criterion = nn.MSELoss()

        # Colleziono tutti i dati di training in CPU una volta sola,
        # così non devo riscorrere il dataloader ad ogni epoch.
        xs_all, ys_all = [], []
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
                y = batch[1] if len(batch) > 1 else None
            else:
                x, y = batch, None
            xs_all.append(x.cpu())
            if y is not None:
                ys_all.append(y.cpu())

        X_all = torch.cat(xs_all, dim=0)   # (n_train, n_features)

        if len(ys_all) == 0:
            raise ValueError(
                "Il dataloader deve restituire (x, y) con le true label. "
                "Assicurati che dataset.labels NON siano state azzerate."
            )
        Y_all = torch.cat(ys_all, dim=0)   # (n_train,)

        for epoch in range(epochs):

            # ----------------------------------------------------------------
            # STEP 1-3: embedding attuali → stima eps → DBSCAN → ARI
            # (solo per monitoraggio, senza gradienti)
            # ----------------------------------------------------------------
            self.eval()
            with torch.no_grad():
                Z_np = self.encoder(X_all.float().to(device)).cpu().numpy()

            # Stima eps sugli embedding grezzi (senza scaling):
            # le k-distanze sono calcolate nello stesso spazio euclideo
            # che userà DBSCAN, quindi eps è perfettamente calibrato.
            eps_auto, _ = self.estimate_eps(Z_np, k=min_samples, percentile=90)

            clustering = DBSCAN(eps=eps_auto, min_samples=min_samples).fit(Z_np)
            pred       = clustering.labels_
            ari        = adjusted_rand_score(Y_all.numpy(), pred)

            # ----------------------------------------------------------------
            # STEP 4-6: surrogate loss + recon loss → backprop
            # ----------------------------------------------------------------
            self.train()
            X = X_all.float().to(device)
            Y = Y_all.to(device)
            Z = self.encoder(X)   # ricalcolo CON grad per il backprop

            if surrogate == 'contrastive':
                surrogate_loss = self.nt_xent_loss(Z, Y, temperature=temperature)
            elif surrogate == 'prototype':
                surrogate_loss = self.prototype_loss(Z, Y, temperature=temperature)
            else:  # fisher
                surrogate_loss = self._fisher_loss(Z, Y, device)

            if recon_weight > 0.0:
                X_hat, _   = self(X)
                recon_loss = recon_criterion(X_hat, X)
            else:
                recon_loss = torch.tensor(0.0, device=device)

            total_loss = surrogate_loss * (1.0 - recon_weight) \
                       + recon_loss    *  recon_weight

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            print(f"Epoch {epoch+1:3d}: "
                  f"eps_auto={eps_auto:.4f} | "
                  f"ARI={ari:.4f} | "
                  f"surrogate={surrogate_loss.item():.6f} | "
                  f"recon={recon_loss.item():.6f} | "
                  f"total={total_loss.item():.6f}")

    # -----------------------------------------------------------------------
    # LOSS DIFFERENZIABILI PER IL CLUSTERING
    # -----------------------------------------------------------------------

    def _fisher_loss(self, Z, Y, device):
        """
        Fisher Discriminant Loss.

        Minimizza la dispersione intra-classe (within) e massimizza quella
        inter-classe (between). Il rapporto within/between è piccolo quando
        i cluster sono compatti e ben separati.
        """
        unique_labels = torch.unique(Y)
        within        = torch.tensor(0.0, device=device)
        between       = torch.tensor(0.0, device=device)
        overall_mean  = Z.mean(dim=0, keepdim=True)
        n_classes     = 0

        for lab in unique_labels:
            mask  = (Y == lab)
            cnt   = mask.sum()
            if cnt <= 0:
                continue
            n_classes += 1
            z_lab  = Z[mask]
            mu_k   = z_lab.mean(dim=0, keepdim=True)
            # scatter intra-classe: somma distanze al centroide di classe
            within  = within  + ((z_lab - mu_k).pow(2).sum()) / (cnt.float() + 1e-8)
            # scatter inter-classe: distanza del centroide di classe dalla media globale
            between = between + (cnt.float() * (mu_k - overall_mean).pow(2).sum())

        if n_classes == 0:
            return torch.tensor(0.0, device=device)

        between = between / float(n_classes)
        return within / (between + 1e-8)

    def nt_xent_loss(self, z, labels, temperature=0.5):
        """
        NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss.

        Per ogni campione i ("ancora"), i campioni della stessa classe
        sono i "positivi" e tutti gli altri sono i "negativi".
        La loss massimizza la similarità coseno con i positivi e la
        minimizza con i negativi, usando la temperatura per scalare i logit.
        """
        device = z.device
        # Normalizza gli embedding sulla sfera unitaria per usare
        # la similarità coseno tramite prodotto scalare
        z_norm   = F.normalize(z, dim=1)
        sim      = torch.matmul(z_norm, z_norm.t()) / (temperature + 1e-8)
        labels   = labels.view(-1)
        N        = z.size(0)
        self_mask = torch.eye(N, dtype=torch.bool, device=device)
        exp_sim  = torch.exp(sim)

        loss_sum = torch.tensor(0.0, device=device)
        valid    = 0

        for i in range(N):
            pos_mask    = (labels == labels[i])
            pos_mask[i] = False   # esclude se stesso dai positivi
            if pos_mask.sum() == 0:
                continue
            numerator   = exp_sim[i][pos_mask].sum()
            denominator = exp_sim[i][~self_mask[i]].sum()
            loss_i      = -torch.log((numerator / (denominator + 1e-12)) + 1e-12)
            loss_sum   += loss_i
            valid      += 1

        if valid == 0:
            return torch.tensor(0.0, device=device)
        return loss_sum / float(valid)

    def prototype_loss(self, z, labels, temperature=0.5):
        """
        Prototype Loss (cross-entropy rispetto ai prototipi di classe).

        Per ogni classe si calcola il "prototipo" come media degli embedding
        dei campioni di quella classe nel batch. Poi si addestra l'encoder
        a classificare ogni campione verso il prototipo della sua classe
        usando una cross-entropy su logit di similarità coseno.

        In pratica: embedding dello stesso device devono essere vicini al
        proprio prototipo e lontani dai prototipi degli altri device.
        """
        labels  = labels.view(-1)
        classes = torch.unique(labels)

        if len(classes) <= 1:
            # Con una sola classe la loss non ha senso
            return torch.tensor(0.0, device=z.device)

        prototypes    = []
        class_indices = torch.empty_like(labels)

        for idx, cls in enumerate(classes):
            mask = (labels == cls)
            class_indices[mask] = idx
            # Prototipo = centroide degli embedding della classe
            prototypes.append(z[mask].mean(dim=0))

        prototypes = torch.stack(prototypes, dim=0)   # (n_classes, emb_size)

        # Similarità coseno tra ogni embedding e ogni prototipo
        z_norm = F.normalize(z,          dim=1)
        p_norm = F.normalize(prototypes, dim=1)
        logits = torch.matmul(z_norm, p_norm.t()) / (temperature + 1e-8)

        # Cross-entropy: ogni campione deve avere logit alto verso il
        # prototipo della sua classe
        return F.cross_entropy(logits, class_indices)