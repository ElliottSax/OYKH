import {
  AbsoluteFill,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
  Sequence,
} from 'remotion';
import {AnimatedFluidCharacter, Pose} from './FluidCharacter';

// Background gradient matching our SVGs
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
        top: 80,
        width: '100%',
        textAlign: 'center',
        fontSize: 72,
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

// Pose label
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
        bottom: 80,
        width: '100%',
        textAlign: 'center',
        fontSize: 48,
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

export const PoseDemo: React.FC = () => {
  // Define pose sequence: each pose lasts ~35 frames (just over 1 second at 30fps)
  const poseSequence: {pose: Pose; startFrame: number; endFrame: number; label: string}[] = [
    {pose: 'standing', startFrame: 0, endFrame: 40, label: 'Standing'},
    {pose: 'waving', startFrame: 40, endFrame: 80, label: 'Waving'},
    {pose: 'pointing', startFrame: 80, endFrame: 120, label: 'Pointing'},
    {pose: 'presenting', startFrame: 120, endFrame: 160, label: 'Presenting'},
    {pose: 'thinking', startFrame: 160, endFrame: 200, label: 'Thinking'},
    {pose: 'celebrating', startFrame: 200, endFrame: 240, label: 'Celebrating'},
    {pose: 'jumping', startFrame: 240, endFrame: 280, label: 'Jumping'},
    {pose: 'walking', startFrame: 280, endFrame: 320, label: 'Walking'},
    {pose: 'standing', startFrame: 320, endFrame: 340, label: 'Standing'},
  ];

  return (
    <AbsoluteFill>
      <Background />

      <Title text="Fluid Character Demo" />

      <AnimatedFluidCharacter
        poses={poseSequence}
        scale={0.55}
        y={30}
      />

      {/* Show pose labels with sequences for animation reset */}
      {poseSequence.map((p, i) => (
        <Sequence key={i} from={p.startFrame} durationInFrames={p.endFrame - p.startFrame}>
          <PoseLabel text={p.label} />
        </Sequence>
      ))}
    </AbsoluteFill>
  );
};
