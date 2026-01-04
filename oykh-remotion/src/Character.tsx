import {Img, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';

export type Pose =
  | 'standing'
  | 'waving'
  | 'pointing'
  | 'presenting'
  | 'thinking'
  | 'celebrating'
  | 'jumping'
  | 'walking';

interface CharacterProps {
  pose: Pose;
  scale?: number;
  x?: number;
  y?: number;
}

export const Character: React.FC<CharacterProps> = ({
  pose,
  scale = 0.5,
  x = 0,
  y = 0,
}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();

  // Subtle idle animation - gentle bobbing
  const bob = Math.sin(frame * 0.1) * 3;

  return (
    <div
      style={{
        position: 'absolute',
        left: '50%',
        top: '50%',
        transform: `translate(-50%, -50%) translate(${x}px, ${y + bob}px) scale(${scale})`,
      }}
    >
      <Img
        src={staticFile(`${pose}.svg`)}
        style={{
          width: 1920,
          height: 1080,
        }}
      />
    </div>
  );
};

interface AnimatedCharacterProps {
  poses: {pose: Pose; startFrame: number; endFrame: number}[];
  scale?: number;
  x?: number;
  y?: number;
}

export const AnimatedCharacter: React.FC<AnimatedCharacterProps> = ({
  poses,
  scale = 0.5,
  x = 0,
  y = 0,
}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();

  // Find current pose based on frame
  let currentPose: Pose = 'standing';
  let nextPose: Pose | null = null;
  let transitionProgress = 0;

  for (let i = 0; i < poses.length; i++) {
    const p = poses[i];
    if (frame >= p.startFrame && frame < p.endFrame) {
      currentPose = p.pose;

      // Check if we're in transition zone (last 10 frames)
      const transitionStart = p.endFrame - 10;
      if (frame >= transitionStart && i < poses.length - 1) {
        nextPose = poses[i + 1].pose;
        transitionProgress = (frame - transitionStart) / 10;
      }
      break;
    }
  }

  // Subtle idle animation
  const bob = Math.sin(frame * 0.1) * 3;

  // Scale bounce on pose change
  const scaleSpring = spring({
    frame: frame % 30,
    fps,
    config: {damping: 12, stiffness: 100},
  });
  const bounceScale = interpolate(scaleSpring, [0, 1], [0.98, 1]);

  return (
    <div
      style={{
        position: 'absolute',
        left: '50%',
        top: '50%',
        transform: `translate(-50%, -50%) translate(${x}px, ${y + bob}px) scale(${scale * bounceScale})`,
      }}
    >
      {/* Current pose */}
      <Img
        src={staticFile(`${currentPose}.svg`)}
        style={{
          width: 1920,
          height: 1080,
          opacity: nextPose ? 1 - transitionProgress : 1,
        }}
      />

      {/* Next pose (for crossfade) */}
      {nextPose && (
        <Img
          src={staticFile(`${nextPose}.svg`)}
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: 1920,
            height: 1080,
            opacity: transitionProgress,
          }}
        />
      )}
    </div>
  );
};
