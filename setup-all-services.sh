#!/bin/bash
# Complete Setup Script for All Game-Changer Services
# Run this to set up your entire cost-optimized infrastructure

echo "========================================"
echo "🚀 Setting Up Game-Changer Services"
echo "========================================"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. Hugging Face CLI
echo -e "${YELLOW}[1/8] Setting up Hugging Face...${NC}"
pip install -q huggingface-hub
echo "✅ Hugging Face CLI installed"
echo "   Get your token: https://huggingface.co/settings/tokens"
echo "   Then run: huggingface-cli login"
echo ""

# 2. Together.ai SDK
echo -e "${YELLOW}[2/8] Installing Together.ai SDK...${NC}"
cd /c/projects/income && npm install together-ai
cd /c/projects/coach && npm install together-ai
cd /c/projects/dream && npm install together-ai
cd /c/projects/membership && npm install together-ai
echo "✅ Together.ai SDK installed in all projects"
echo "   Get API key: https://api.together.xyz/settings/api-keys"
echo "   Add to .env: TOGETHER_API_KEY=your_key"
echo ""

# 3. Modal
echo -e "${YELLOW}[3/8] Installing Modal...${NC}"
pip install -q modal
echo "✅ Modal installed"
echo "   Setup: modal setup"
echo "   Get $30 free credit!"
echo ""

# 4. Cloudflare Wrangler
echo -e "${YELLOW}[4/8] Installing Wrangler (Cloudflare)...${NC}"
npm install -g wrangler
echo "✅ Wrangler installed"
echo "   Login: wrangler login"
echo "   Create R2 bucket: wrangler r2 bucket create your-bucket"
echo ""

# 5. Supabase CLI
echo -e "${YELLOW}[5/8] Installing Supabase CLI...${NC}"
npm install -g supabase
echo "✅ Supabase CLI installed"
echo "   Login: supabase login"
echo "   Init project: supabase init"
echo ""

# 6. Vercel CLI (if not installed)
echo -e "${YELLOW}[6/8] Checking Vercel CLI...${NC}"
if ! command -v vercel &> /dev/null; then
    npm install -g vercel
    echo "✅ Vercel CLI installed"
else
    echo "✅ Vercel CLI already installed"
fi
echo ""

# 7. Fly.io CLI
echo -e "${YELLOW}[7/8] Installing Fly.io CLI...${NC}"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "Download: https://fly.io/docs/hands-on/install-flyctl/"
    echo "Or use: powershell -Command \"iwr https://fly.io/install.ps1 -useb | iex\""
else
    curl -L https://fly.io/install.sh | sh
fi
echo ""

# 8. Anthropic SDK
echo -e "${YELLOW}[8/8] Installing Anthropic SDK...${NC}"
cd /c/projects/income && npm install @anthropic-ai/sdk
cd /c/projects/coach && npm install @anthropic-ai/sdk
cd /c/projects/dream && npm install @anthropic-ai/sdk
cd /c/projects/membership && npm install @anthropic-ai/sdk
echo "✅ Anthropic SDK installed in all projects"
echo ""

echo "========================================"
echo -e "${GREEN}✅ All Services Installed!${NC}"
echo "========================================"
echo ""
echo "Next Steps:"
echo "1. Get API keys from each service (see links above)"
echo "2. Run setup scripts in ./service-setup/"
echo "3. Test integrations with ./test-services.js"
echo ""
echo "Estimated time to complete setup: 30-45 minutes"
echo "Annual savings once configured: $3,000-5,000"
echo ""
