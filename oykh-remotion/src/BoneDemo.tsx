import {AbsoluteFill, Sequence, useCurrentFrame, useVideoConfig, spring} from 'remotion';
import {AnimatedUnifiedBoneCharacter, POSES} from './UnifiedBoneCharacter';

// Background gradient matching our style
const Background: React.FC = () => (
  <AbsoluteFill
    style={{
      background: 'linear-gradient(135deg, #45b8d1 0%, #6a6acd 30%, #c080a8 60%, #e8a060 100%)',
    }}
  />
);

// Title text component
const Title: React.FC<{text: string}> = ({text}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();

  const opacity = spring({
    frame,
    fps,
    config: {damping: 20},
  });

  return (
    <div
      style={{
        position: 'absolute',
        top: 60,
        width: '100%',
        textAlign: 'center',
        fontSize: 64,
        fontFamily: 'Arial, sans-serif',
        fontWeight: 'bold',
        color: 'white',
        textShadow: '4px 4px 8px rgba(0,0,0,0.3)',
        opacity,
      }}
    >
      {text}
    </div>
  );
};

// Pose label that animates in
const PoseLabel: React.FC<{text: string}> = ({text}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();

  const scale = spring({
    frame,
    fps,
    config: {damping: 12, stiffness: 100},
  });

  return (
    <div
      style={{
        position: 'absolute',
        bottom: 60,
        width: '100%',
        textAlign: 'center',
        fontSize: 42,
        fontFamily: 'Arial, sans-serif',
        fontWeight: 'bold',
        color: 'white',
        textShadow: '3px 3px 6px rgba(0,0,0,0.3)',
        transform: `scale(${scale})`,
      }}
    >
      {text}
    </div>
  );
};

export const BoneDemo: React.FC = () => {
  const {fps} = useVideoConfig();

  // Define pose sequence with durations in seconds
  const poseSequence: {pose: keyof typeof POSES; duration: number; label: string}[] = [
    {pose: 'standing', duration: 1.5, label: 'Standing'},
    {pose: 'waving', duration: 1.5, label: 'Waving'},
    {pose: 'pointing', duration: 1.5, label: 'Pointing'},
    {pose: 'presenting', duration: 1.5, label: 'Presenting'},
    {pose: 'thinking', duration: 1.5, label: 'Thinking'},
    {pose: 'celebrating', duration: 1.5, label: 'Celebrating'},
    {pose: 'jumping', duration: 1.5, label: 'Jumping'},
    {pose: 'walking1', duration: 0.5, label: 'Walking'},
    {pose: 'walking2', duration: 0.5, label: 'Walking'},
    {pose: 'walking1', duration: 0.5, label: 'Walking'},
    {pose: 'walking2', duration: 0.5, label: 'Walking'},
    {pose: 'standing', duration: 1, label: 'Standing'},
  ];

  // Calculate frame ranges for labels
  let currentFramePos = 0;
  const labelSequences: {label: string; startFrame: number; durationFrames: number}[] = [];

  for (const item of poseSequence) {
    const durationFrames = Math.round(item.duration * fps);
    labelSequences.push({
      label: item.label,
      startFrame: currentFramePos,
      durationFrames,
    });
    currentFramePos += durationFrames;
  }

  return (
    <AbsoluteFill>
      <Background />

      <Title text="Cel-Shaded Bone Demo" />

      <AnimatedUnifiedBoneCharacter
        poseSequence={poseSequence}
        scale={0.85}
        y={480}
      />

      {/* Show pose labels */}
      {labelSequences.map((seq, i) => (
        <Sequence key={i} from={seq.startFrame} durationInFrames={seq.durationFrames}>
          <PoseLabel text={seq.label} />
        </Sequence>
      ))}
    </AbsoluteFill>
  );
};
