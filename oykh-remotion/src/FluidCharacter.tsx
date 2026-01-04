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

interface AnimatedFluidCharacterProps {
  poses: {pose: Pose; startFrame: number; endFrame: number}[];
  scale?: number;
  x?: number;
  y?: number;
}

export const AnimatedFluidCharacter: React.FC<AnimatedFluidCharacterProps> = ({
  poses,
  scale = 0.5,
  x = 0,
  y = 0,
}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();

  // Find current pose and transition info
  let currentPose: Pose = 'standing';
  let nextPose: Pose | null = null;
  let transitionProgress = 0;
  const transitionDuration = 25; // Increased duration for more complex transition

  for (let i = 0; i < poses.length; i++) {
    const p = poses[i];
    if (frame >= p.startFrame && frame < p.endFrame) {
      currentPose = p.pose;

      const transitionStart = p.endFrame - transitionDuration;
      if (frame >= transitionStart && i < poses.length - 1) {
        nextPose = poses[i + 1].pose;
        transitionProgress = (frame - transitionStart) / transitionDuration;
      }
      break;
    }
  }

  // -- Animation Principles --

  // 1. Idle Animation (Subtle Bobbing)
  const bob = Math.sin(frame * 0.1) * 3;

  // 2. Anticipation and Overshoot using a spring
  const transitionSpring = spring({
    frame: transitionProgress * fps,
    fps,
    config: {
      damping: 15,    // Lower damping creates more bounce (overshoot)
      stiffness: 100,
      mass: 0.8,
    },
  });

  // 2a. Anticipation: Character squashes down before moving
  const anticipationY = interpolate(
    transitionProgress,
    [0, 0.2, 1],
    [0, 15, 0] // Dips down 15px at the start of the transition
  );
  const anticipationScaleY = interpolate(
    transitionProgress,
    [0, 0.2, 1],
    [1, 0.95, 1] // Squashes to 95% height
  );
  const anticipationScaleX = interpolate(
    transitionProgress,
    [0, 0.2, 1],
    [1, 1.05, 1] // Stretches to 105% width
  );
  
  // 3. Arcing Motion: Character moves in an arc during transition
  const arcY = Math.sin(transitionProgress * Math.PI) * -30; // Moves up 30px in an arc

  // 4. Overshoot: The spring will naturally overshoot the final value
  const finalScale = interpolate(transitionSpring, [0, 1], [1, 1]); // The spring handles the bounce

  const finalTransform = `
    translate(-50%, -50%)
    translate(${x}px, ${y + bob + anticipationY + arcY}px)
    scale(${scale * finalScale})
    scaleX(${anticipationScaleX})
    scaleY(${anticipationScaleY})
  `;

  return (
    <div
      style={{
        position: 'absolute',
        left: '50%',
        top: '50%',
        transform: finalTransform,
      }}
    >
      {/* Current pose */}
      <Img
        src={staticFile(`${currentPose}.svg`)}
        style={{
          width: 1920,
          height: 1080,
          opacity: nextPose ? interpolate(transitionProgress, [0, 0.5, 1], [1, 0, 0]) : 1,
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
            opacity: interpolate(transitionProgress, [0, 0.5, 1], [0, 1, 1]),
          }}
        />
      )}
    </div>
  );
};
