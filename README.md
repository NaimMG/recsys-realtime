# 🎯 Real-Time Recommendation System

Moteur de recommandation en temps réel basé sur une architecture event-driven,
déployé entièrement sur Azure avec des services managés.

## 🏗️ Architecture
```
Retail Rocket Events
      ↓
Azure Event Hubs (Kafka)
      ↓
Consumer Bytewax (stream processing)
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
| Stream processing | Bytewax |
| Modèle offline | ALS via Implicit |
| Feature store | Redis |
| API | FastAPI |
| Déploiement | Azure Container Apps |
| CI/CD | GitHub Actions + Azure DevOps |
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

## 📊 Métriques

- Latence p50 : < 20ms
- Latence p99 : < 50ms
- Throughput : > 1000 req/sec