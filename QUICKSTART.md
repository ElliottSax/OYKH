# 🚀 OYKH Quick Start Guide

Get your viral video generator running in **5 minutes**.

---

## Step 1: Get Your Gemini API Key (2 min)

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Click "Create API Key"
3. Copy the key (starts with `AIza...`)

---

## Step 2: Setup Project (1 min)

```bash
# Clone/navigate to project
cd /c/projects/oykh-temp

# Install dependencies
npm install

# Create environment file
cp .env.example .env.local

# Open .env.local and paste your API key
# GEMINI_API_KEY=AIzaSy...your_key_here
```

---

## Step 3: Run Development Server (1 min)

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

---

## Step 4: Generate Your First Video (1 min)

1. Enter a topic (e.g., "Why Do We Dream?")
2. Click "HOOK IT"
3. Select a hook
4. Watch the magic happen! ✨

**Note**: Images and audio are placeholders in dev mode. See [README.md](README.md) for production setup.

---

## What You'll Get

### Development Mode (Now):
- ✅ Viral script generation (real Gemini API)
- ✅ 150-180 shot breakdown
- ✅ Retention optimization
- ✅ Mock visual placeholders
- ✅ Mock audio
- **Cost**: ~$0.002 per video

### Production Mode (After Backend Setup):
- ✅ Real Imagen 3 character animation
- ✅ Real Google TTS narration
- ✅ MP4 video export
- ✅ Professional quality
- **Cost**: ~$2 per video (with optimization)

---

## Next Steps

1. **Try It Out**: Generate a few test videos
2. **Review Output**: Check `IMPLEMENTATION_SUMMARY.md`
3. **Go Production**: Follow [README.md](README.md) Phase 1-3

---

## Troubleshooting

**"Model not found"**:
- Check your API key is correct
- Make sure it starts with `AIza`

**Nothing happens when clicking buttons**:
- Check browser console (F12)
- Verify API key in `.env.local`

**Want real images/audio**:
- See [README.md](README.md) → Production Setup

---

## Support

- 📚 Full docs: [README.md](README.md)
- 🔍 Implementation details: [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
- 🐛 Issues: Check console logs

---

**Ready?** → `npm run dev` and go to http://localhost:3000

Let's make some viral videos! 🎬
