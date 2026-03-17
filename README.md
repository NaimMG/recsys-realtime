# 🎯 Real-Time Recommendation System

Moteur de recommandation en temps réel basé sur une architecture event-driven,
déployé entièrement sur Azure avec des services managés.

## 🏗️ Architecture
```
Retail Rocket Events
      ↓
Azure Event Hubs (Kafka)
      ↓
Consumer (stream processing)
      ↓
Redis (embeddings + sessions)
      ↓
FastAPI /recommend/{userId}
      ↓
Grafana Dashboard (monitoring live)
```

## 🛠️ Stack technique

| Composant | Technologie |
|---|---|
| Streaming | Azure Event Hubs (Kafka compatible) |
| Stream processing | Consumer Python |
| Modèle offline | ALS via Implicit |
| Feature store | Redis |
| API | FastAPI |
| Déploiement | Azure Container Apps |
| CI/CD | GitHub Actions |
| Monitoring | Prometheus + Grafana |

## 📦 Datasets

- Retail Rocket — événements e-commerce réels (views, addtocart, transactions)
- MovieLens 1M — interactions utilisateur/film pour le collaborative filtering

## 🚀 Lancer le projet

### Prérequis
- Compte Azure avec les ressources provisionnées
- Variables d'environnement configurées (voir .env.example)

### Déploiement
```bash
cp .env.example .env
# Remplir les valeurs dans .env
bash scripts/setup_azure.sh
```

## 🌐 API Live

L'API est déployée et accessible publiquement :

- **Endpoint** : `http://51.103.26.118:8000`
- **Documentation** : `http://51.103.26.118:8000/docs`
- **Métriques** : `http://51.103.26.118:8000/metrics`
- **Grafana** : `http://51.103.26.118:3000`

### Exemple
```bash
curl http://51.103.26.118:8000/recommend/257597
```

### Réponse
```json
{
  "user_id": 257597,
  "recommendations": [355908, 317948, 393910, 463851, 44821, 308985, 145089, 140705, 24110, 185570],
  "strategy": "als+session",
  "latency_ms": 21.02
}
```

## 📊 Métriques

- Latence p50 : < 20ms
- Latence p99 : < 50ms
- Throughput : > 1000 req/sec