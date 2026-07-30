const fs = require('fs');
const content = fs.readFileSync('C:/projects/oykh-temp/test-procrastination-optimized-wrapped.json', 'utf8');
const data = JSON.parse(content);
const script = data.script || data;

if (!script.chapters) {
  console.error('Error: No chapters found in script');
  console.log('Script keys:', Object.keys(script));
  process.exit(1);
}

script.chapters.forEach(chapter => {
  chapter.shots.forEach(shot => {
    const orig = shot.imagenPrompt;

    // Extract key elements only
    let action = '';
    if (orig.includes('pointing')) action = 'pointing gesture';
    else if (orig.includes('spread')) action = 'arms spread';
    else if (orig.includes('thinking') || orig.includes('temple')) action = 'thinking pose';
    else if (orig.includes('holding coffee')) action = 'holding coffee';
    else if (orig.includes('clock')) action = 'holding clock';
    else if (orig.includes('frustrated')) action = 'frustrated pose';
    else if (orig.includes('looking down')) action = 'looking down';
    else if (orig.includes('excited')) action = 'excited pose';
    else if (orig.includes('relaxed')) action = 'relaxed pose';
    else action = 'standing';

    // Extract prop
    let prop = '';
    if (orig.includes('lightbulb')) prop = ', lightbulb';
    else if (orig.includes('coffee') && !action.includes('coffee')) prop = ', coffee mug';
    else if (orig.includes('brain')) prop = ', brain icon';
    else if (orig.includes('clock') && !action.includes('clock')) prop = ', clock';
    else if (orig.includes('checkmark')) prop = ', checkmark';

    // Ultra-minimal format (25-30 words)
    shot.imagenPrompt = `White stick figure, round head, dot eyes, ${action}${prop}. Blue-purple gradient. Kurzgesagt educational style.`;
  });
});

fs.writeFileSync('C:/projects/oykh-temp/test-procrastination-ultra-minimal.json', JSON.stringify(script, null, 2));
console.log('✅ Ultra-minimal script created');
console.log('Sample prompt:', script.chapters[0].shots[0].imagenPrompt);
