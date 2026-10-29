# Painel IA — Bolsa Americana (simulação)

Protótipo educativo de painel web para acompanhar ações/ETFs dos EUA, visualizar indicadores e testar uma simulação sem dinheiro real.

## Requisitos
- Python 3.10 ou superior
- Internet para buscar dados históricos

## Instalação
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
Abra o endereço local mostrado no terminal. Para acessar pelo celular, o computador e o celular precisam estar na mesma rede e o app deve ser iniciado com configuração de rede adequada; para acesso fora de casa, será necessário publicar o app em um serviço confiável.

## O que faz
- Busca dados históricos do símbolo selecionado.
- Calcula indicadores simples.
- Treina um Random Forest para estimar se o preço estará acima do atual após 5, 10 ou 20 pregões.
- Mostra uma avaliação cronológica básica.
- Simula uma estratégia de médias móveis com saldo virtual.

## Limitações importantes
- Não envia ordens e não conecta a contas de corretoras.
- Dados podem estar atrasados, incompletos ou sujeitos a termos de uso do provedor.
- Acurácia histórica não equivale a rentabilidade.
- A simulação não inclui todos os custos, impostos, spreads, slippage ou eventos corporativos.
- O modelo não garante previsões corretas. Não use este protótipo como recomendação financeira.

## Próximos passos
1. Adicionar validação walk-forward rigorosa.
2. Incluir custos e slippage na simulação.
3. Persistir histórico de previsões e resultados em banco de dados.
4. Adicionar alertas informativos, sem execução de ordens.
5. Publicar com autenticação e proteção de segredos.
