# Optimized Prompt Strategy

## Key Finding

**Coffee video (simple prompts) = Better consistency than Procrastination (complex prompts)**

## Why Simple Works Better

### Coffee's Winning Formula (40-50 words):

```
White stick figure with thick black vector outline (8-10px stroke),
perfectly round head, two simple black dot eyes,
[natural action description],
[props if any].
Blue-purple gradient background with sophisticated lighting.
```

### Why It Works:

1. **Natural Language** - AI understands "pointing at viewer" better than "190px diameter, 45px apart"
2. **Concise** - 40-50 words vs 100-120 words = less confusion
3. **Positive Framing** - Describes what TO include, not what to avoid
4. **Flowing Narrative** - Reads like a scene description, not a parts list
5. **Minimal Tech Specs** - Only mentions "8-10px stroke", rest is descriptive

## What Made Procrastination Worse

### Overly Complex Formula (100-120 words):

```
Character in '[POSE]' pose. [PROPS].
Character is pure white stick figure with perfect geometric circle head (190px diameter),
two black dot eyes (12px, 45px apart), black 10px outline on head only.
Simple stick body with black lines (10px width) for torso, arms, legs.
[Detailed prop specs with exact hex codes]
Background is smooth linear gradient from deep purple (#4A148C) to medium purple (#7B1FA2).
Kurzgesagt/TED-Ed educational video style. Ultra-minimalist vector illustration.
NO textures, NO shadows on character, NO gradients on character/props, NO complex details, NO realistic elements.
```

### Why It Failed:

1. **Over-specification** - Too many exact measurements confuse the AI
2. **Robotic Structure** - Reads like technical documentation
3. **Negative Prompts** - "NO textures, NO shadows" can backfire
4. **Information Overload** - Each constraint can conflict with others
5. **Lost the Flow** - Compartmentalized instead of integrated

## Optimized Prompt Template

### Character Base (use for every shot):

```
White stick figure with thick black vector outline, perfectly round head, two simple black dot eyes
```

### Action (natural language):

```
pointing at viewer with inviting gesture
holding a coffee mug and looking thoughtful
spreading arms wide in explanation
touching temple in thinking pose
```

### Props (simple descriptions):

```
, coffee mug in hand with steam lines
, brain icon floating above head
, lightbulb appearing above
, holding a simple clock
```

### Background (consistent):

```
Blue-purple gradient background with sophisticated lighting
```

### Style Hint (optional, at end):

```
Educational vector illustration, Kurzgesagt style
```

## Full Example Prompts

**Pointing shot:**

```
White stick figure with thick black vector outline, perfectly round head, two simple black dot eyes, pointing at viewer with inviting gesture, lightbulb appearing above. Blue-purple gradient background with sophisticated lighting. Educational vector illustration, Kurzgesagt style.
```

**Thinking shot:**

```
White stick figure with thick black vector outline, perfectly round head, two simple black dot eyes, hand touching temple in thoughtful pose. Blue-purple gradient background with sophisticated lighting. Educational vector illustration, Kurzgesagt style.
```

**Holding prop shot:**

```
White stick figure with thick black vector outline, perfectly round head, two simple black dot eyes, holding coffee mug with steam lines rising. Blue-purple gradient background with sophisticated lighting. Educational vector illustration, Kurzgesagt style.
```

## Implementation

Replace STYLE_LOCK with simple template that concatenates:

1. Character base (constant)
2. Natural action from imagenPrompt
3. Background (constant)
4. Style hint (constant)

Keep it under 50 words, natural language, positive framing only.
