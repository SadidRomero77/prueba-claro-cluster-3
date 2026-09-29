# Despliegue en AWS (EC2 + Docker + Caddy + Cloudflare + SSM)

Mismo patrón de Kibo: una instancia EC2 pequeña con Docker, Caddy para HTTPS y usuario/clave,
DNS en Cloudflare y acceso por SSM (sin abrir el puerto 22).

## 1. Instancia

| Recurso | Valor sugerido |
|---|---|
| AMI | Amazon Linux 2023 |
| Tipo | t3.medium (2 vCPU, 4 GB) — LightGBM + SHAP + Streamlit |
| Disco | 20 GB gp3 |
| Security group | Entrada 80 y 443 desde 0.0.0.0/0 (o solo IPs de Cloudflare). Sin 22. |
| IAM role | `AmazonSSMManagedInstanceCore` |
| Región | us-east-1 |

User data (instala Docker y compose):

```bash
#!/bin/bash
dnf install -y docker git
systemctl enable --now docker
mkdir -p /usr/local/lib/docker/cli-plugins
curl -sL https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64 \
  -o /usr/local/lib/docker/cli-plugins/docker-compose
chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
```

## 2. DNS en Cloudflare

Registro `A` → IP elástica de la instancia, por ejemplo `cluster3.<tu-dominio>`.
Para que Caddy emita el certificado, deja el proxy de Cloudflare en **DNS only** la primera vez;
después puedes activarlo con SSL en modo **Full (strict)**.

## 3. Código y artefactos

```bash
aws ssm start-session --target <instance-id>
sudo -iu ec2-user
git clone <repo> claro-cluster3 && cd claro-cluster3
```

Los artefactos (no están en git) se generan en local con `uv run cluster3` y se suben a la instancia:

```bash
# en local
tar czf artefactos.tgz outputs models data/processed data/labels
aws s3 cp artefactos.tgz s3://<bucket-temporal>/
# en la instancia
aws s3 cp s3://<bucket-temporal>/artefactos.tgz . && tar xzf artefactos.tgz && rm artefactos.tgz
aws s3 rm s3://<bucket-temporal>/artefactos.tgz
```

`data/raw` nunca se sube a la instancia.

## 4. Variables y arranque

```bash
cp .env.example .env   # ANTHROPIC_API_KEY o OPENAI_API_KEY, LLM_PROVIDER
cat >> .env <<EOF
APP_DOMAIN=cluster3.<tu-dominio>
APP_USER=claro
APP_PASSWORD_HASH='<hash de caddy hash-password>'
EOF
cd deploy/aws && docker compose --env-file ../../.env up -d --build
docker compose logs -f app
```

El hash bcrypt lleva `$`: déjalo entre comillas simples para que Docker Compose no lo interprete.

Salud: `curl -s https://cluster3.<tu-dominio>/_stcore/health` (con usuario/clave).

## 5. Cierre del proceso (datos internos de Claro)

```bash
docker compose down -v
cd ~ && rm -rf claro-cluster3
```

Después: terminar la instancia, liberar la IP elástica, borrar el registro DNS y el bucket temporal.
