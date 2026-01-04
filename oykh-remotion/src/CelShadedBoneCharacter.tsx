import {useCurrentFrame, useVideoConfig} from 'remotion';

interface Pose {
  rightShoulder: number;  // 0 = down, 90 = horizontal, 180 = up
  rightElbow: number;     // 0 = straight, positive = bend
  leftShoulder: number;
  leftElbow: number;
  rightHip: number;       // 0 = down, positive = forward
  rightKnee: number;
  leftHip: number;
  leftKnee: number;
}

export const POSES: Record<string, Pose> = {
  standing: {
    rightShoulder: 12, rightElbow: 25,
    leftShoulder: 12, leftElbow: 25,
    rightHip: 0, rightKnee: 0,
    leftHip: 0, leftKnee: 0,
  },
  waving: {
    rightShoulder: 155, rightElbow: 40,
    leftShoulder: 12, leftElbow: 25,
    rightHip: 0, rightKnee: 0,
    leftHip: 0, leftKnee: 0,
  },
  pointing: {
    rightShoulder: 90, rightElbow: 0,
    leftShoulder: 12, leftElbow: 25,
    rightHip: 0, rightKnee: 0,
    leftHip: 0, leftKnee: 0,
  },
  presenting: {
    rightShoulder: 125, rightElbow: 15,
    leftShoulder: 12, leftElbow: 25,
    rightHip: 0, rightKnee: 0,
    leftHip: 0, leftKnee: 0,
  },
  thinking: {
    rightShoulder: 115, rightElbow: 155,
    leftShoulder: 12, leftElbow: 25,
    rightHip: 0, rightKnee: 0,
    leftHip: 0, leftKnee: 0,
  },
  celebrating: {
    rightShoulder: 165, rightElbow: 25,
    leftShoulder: 165, leftElbow: 25,
    rightHip: 0, rightKnee: 0,
    leftHip: 0, leftKnee: 0,
  },
  jumping: {
    rightShoulder: 140, rightElbow: 10,
    leftShoulder: 140, leftElbow: 10,
    rightHip: 18, rightKnee: 28,
    leftHip: 18, leftKnee: 28,
  },
  walking1: {
    rightShoulder: 25, rightElbow: 35,
    leftShoulder: 5, leftElbow: 15,
    rightHip: -12, rightKnee: 8,
    leftHip: 22, leftKnee: 32,
  },
  walking2: {
    rightShoulder: 5, rightElbow: 15,
    leftShoulder: 25, leftElbow: 35,
    rightHip: 22, rightKnee: 32,
    leftHip: -12, leftKnee: 8,
  },
};

function lerp(a: number, b: number, t: number) { return a + (b - a) * t; }

function interpolatePose(from: Pose, to: Pose, t: number): Pose {
  return {
    rightShoulder: lerp(from.rightShoulder, to.rightShoulder, t),
    rightElbow: lerp(from.rightElbow, to.rightElbow, t),
    leftShoulder: lerp(from.leftShoulder, to.leftShoulder, t),
    leftElbow: lerp(from.leftElbow, to.leftElbow, t),
    rightHip: lerp(from.rightHip, to.rightHip, t),
    rightKnee: lerp(from.rightKnee, to.rightKnee, t),
    leftHip: lerp(from.leftHip, to.leftHip, t),
    leftKnee: lerp(from.leftKnee, to.leftKnee, t),
  };
}

// Get point at angle (0=down, 90=right, 180=up, 270=left)
// In SVG: Y increases downward, so we use (90 - angle) to map correctly
function pt(x: number, y: number, angle: number, dist: number) {
  const rad = (90 - angle) * Math.PI / 180;
  return { x: x + Math.cos(rad) * dist, y: y + Math.sin(rad) * dist };
}

// Create a capsule (pill) shape path between two points
function capsule(x1: number, y1: number, x2: number, y2: number, r: number): string {
  const dx = x2 - x1, dy = y2 - y1;
  const len = Math.sqrt(dx * dx + dy * dy) || 1;
  const nx = (-dy / len) * r;  // perpendicular x offset
  const ny = (dx / len) * r;   // perpendicular y offset

  return `
    M ${x1 + nx} ${y1 + ny}
    L ${x2 + nx} ${y2 + ny}
    A ${r} ${r} 0 0 1 ${x2 - nx} ${y2 - ny}
    L ${x1 - nx} ${y1 - ny}
    A ${r} ${r} 0 0 1 ${x1 + nx} ${y1 + ny}
    Z
  `;
}

interface CelShadedBoneCharacterProps {
  pose: Pose;
  x?: number;
  y?: number;
  scale?: number;
}

export const CelShadedBoneCharacter: React.FC<CelShadedBoneCharacterProps> = ({
  pose,
  x = 960,
  y = 540,
  scale = 1,
}) => {
  // Dimensions
  const headR = 115;
  const shoulderY = 140;
  const shoulderX = 40;

  const upperArmLen = 95;
  const lowerArmLen = 90;
  const armR = 32;  // radius (half-width) of arm capsule
  const handR = 34;

  const bodyTop = 120;
  const bodyBottom = 340;
  const bodyWidth = 65;

  const hipY = 320;
  const hipX = 30;
  const upperLegLen = 125;
  const lowerLegLen = 120;
  const legR = 33;  // radius (half-width) of leg capsule
  const footR = 32;

  // Calculate joint positions
  const rShoulder = { x: shoulderX, y: shoulderY };
  const lShoulder = { x: -shoulderX, y: shoulderY };

  // Right arm: angle 0=down, positive=toward right/up
  const rElbow = pt(rShoulder.x, rShoulder.y, pose.rightShoulder, upperArmLen);
  const rHand = pt(rElbow.x, rElbow.y, pose.rightShoulder + pose.rightElbow, lowerArmLen);

  // Left arm: mirror by negating angle (goes toward left instead of right)
  const lElbow = pt(lShoulder.x, lShoulder.y, -pose.leftShoulder, upperArmLen);
  const lHand = pt(lElbow.x, lElbow.y, -pose.leftShoulder - pose.leftElbow, lowerArmLen);

  const rHip = { x: hipX, y: hipY };
  const lHip = { x: -hipX, y: hipY };

  // Right leg: negative angle so 0=down, positive hip=forward
  const rKnee = pt(rHip.x, rHip.y, -pose.rightHip, upperLegLen);
  const rFoot = pt(rKnee.x, rKnee.y, -pose.rightHip + pose.rightKnee, lowerLegLen);

  // Left leg: same as right (both point down)
  const lKnee = pt(lHip.x, lHip.y, -pose.leftHip, upperLegLen);
  const lFoot = pt(lKnee.x, lKnee.y, -pose.leftHip + pose.leftKnee, lowerLegLen);

  // Common style props
  const fill = "url(#cel-grad)";
  const stroke = "#202038";
  const strokeWidth = 12;

  return (
    <svg width={1920} height={1080} style={{ position: 'absolute', left: 0, top: 0 }}>
      <defs>
        <linearGradient id="cel-grad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#FFFFFF" />
          <stop offset="50%" stopColor="#F5F5FA" />
          <stop offset="51%" stopColor="#E8E8F0" />
          <stop offset="100%" stopColor="#D8D8E0" />
        </linearGradient>

        <filter id="hard-shadow" x="-50%" y="-50%" width="200%" height="200%">
            <feOffset dx="8" dy="10" in="SourceAlpha" result="offset" />
            <feFlood floodColor="#1a1a2e" floodOpacity="0.6" result="color" />
            <feComposite in="color" in2="offset" operator="in" result="shadow" />
            <feMerge>
                <feMergeNode in="shadow" />
                <feMergeNode in="SourceGraphic" />
            </feMerge>
        </filter>

        <clipPath id="headClip">
          <circle cx="0" cy="0" r={headR}/>
        </clipPath>
      </defs>

      <g transform={`translate(${x}, ${y}) scale(${scale})`}>
        {/* Ground shadow */}
        <ellipse cx="0" cy={hipY + upperLegLen + lowerLegLen + 50} rx="150" ry="20" fill="rgba(30, 30, 50, 0.2)" />

        <g filter="url(#hard-shadow)">
          {/* === ALL FILLS FIRST (creates connected appearance) === */}
          {/* Legs fills */}
          <path d={capsule(rHip.x, rHip.y, rKnee.x, rKnee.y, legR)} fill={fill} stroke="none"/>
          <path d={capsule(rKnee.x, rKnee.y, rFoot.x, rFoot.y, legR * 0.9)} fill={fill} stroke="none"/>
          <circle cx={rFoot.x} cy={rFoot.y} r={footR} fill={fill} stroke="none"/>
          <path d={capsule(lHip.x, lHip.y, lKnee.x, lKnee.y, legR)} fill={fill} stroke="none"/>
          <path d={capsule(lKnee.x, lKnee.y, lFoot.x, lFoot.y, legR * 0.9)} fill={fill} stroke="none"/>
          <circle cx={lFoot.x} cy={lFoot.y} r={footR} fill={fill} stroke="none"/>

          {/* Body fill */}
          <ellipse cx="0" cy={(bodyTop + bodyBottom) / 2} rx={bodyWidth} ry={(bodyBottom - bodyTop) / 2} fill={fill} stroke="none"/>

          {/* Arms fills */}
          <path d={capsule(lShoulder.x, lShoulder.y, lElbow.x, lElbow.y, armR)} fill={fill} stroke="none"/>
          <path d={capsule(lElbow.x, lElbow.y, lHand.x, lHand.y, armR * 0.88)} fill={fill} stroke="none"/>
          <circle cx={lHand.x} cy={lHand.y} r={handR} fill={fill} stroke="none"/>
          <path d={capsule(rShoulder.x, rShoulder.y, rElbow.x, rElbow.y, armR)} fill={fill} stroke="none"/>
          <path d={capsule(rElbow.x, rElbow.y, rHand.x, rHand.y, armR * 0.88)} fill={fill} stroke="none"/>
          <circle cx={rHand.x} cy={rHand.y} r={handR} fill={fill} stroke="none"/>

          {/* Head fill */}
          <circle cx="0" cy="0" r={headR} fill={fill} stroke="none"/>

          {/* === NOW STROKES (on top of fills) === */}
          {/* Legs strokes */}
          <path d={capsule(rHip.x, rHip.y, rKnee.x, rKnee.y, legR)}
                fill="none" stroke={stroke} strokeWidth={strokeWidth} strokeLinejoin="round" strokeLinecap="round"/>
          <path d={capsule(rKnee.x, rKnee.y, rFoot.x, rFoot.y, legR * 0.9)}
                fill="none" stroke={stroke} strokeWidth={strokeWidth} strokeLinejoin="round" strokeLinecap="round"/>
          <circle cx={rFoot.x} cy={rFoot.y} r={footR} fill="none" stroke={stroke} strokeWidth={strokeWidth}/>
          <path d={capsule(lHip.x, lHip.y, lKnee.x, lKnee.y, legR)}
                fill="none" stroke={stroke} strokeWidth={strokeWidth} strokeLinejoin="round" strokeLinecap="round"/>
          <path d={capsule(lKnee.x, lKnee.y, lFoot.x, lFoot.y, legR * 0.9)}
                fill="none" stroke={stroke} strokeWidth={strokeWidth} strokeLinejoin="round" strokeLinecap="round"/>
          <circle cx={lFoot.x} cy={lFoot.y} r={footR} fill="none" stroke={stroke} strokeWidth={strokeWidth}/>

          {/* Body stroke */}
          <ellipse cx="0" cy={(bodyTop + bodyBottom) / 2} rx={bodyWidth} ry={(bodyBottom - bodyTop) / 2}
                   fill="none" stroke={stroke} strokeWidth={strokeWidth}/>

          {/* Arms strokes */}
          <path d={capsule(lShoulder.x, lShoulder.y, lElbow.x, lElbow.y, armR)}
                fill="none" stroke={stroke} strokeWidth={strokeWidth} strokeLinejoin="round" strokeLinecap="round"/>
          <path d={capsule(lElbow.x, lElbow.y, lHand.x, lHand.y, armR * 0.88)}
                fill="none" stroke={stroke} strokeWidth={strokeWidth} strokeLinejoin="round" strokeLinecap="round"/>
          <circle cx={lHand.x} cy={lHand.y} r={handR} fill="none" stroke={stroke} strokeWidth={strokeWidth}/>
          <path d={capsule(rShoulder.x, rShoulder.y, rElbow.x, rElbow.y, armR)}
                fill="none" stroke={stroke} strokeWidth={strokeWidth} strokeLinejoin="round" strokeLinecap="round"/>
          <path d={capsule(rElbow.x, rElbow.y, rHand.x, rHand.y, armR * 0.88)}
                fill="none" stroke={stroke} strokeWidth={strokeWidth} strokeLinejoin="round" strokeLinecap="round"/>
          <circle cx={rHand.x} cy={rHand.y} r={handR} fill="none" stroke={stroke} strokeWidth={strokeWidth}/>

          {/* Head stroke */}
          <circle cx="0" cy="0" r={headR} fill="none" stroke={stroke} strokeWidth={strokeWidth}/>
          
          {/* Minimalist Shine */}
          <g clipPath="url(#headClip)">
            <path
              d="M -90 -80 A 100 100 0 0 1 50 -50"
              stroke="white"
              strokeWidth="20"
              fill="none"
              strokeLinecap="round"
              opacity="0.6"
            />
          </g>

          {/* Eyes */}
          <ellipse cx="-28" cy="5" rx="12" ry="15" fill="#202038"/>
          <ellipse cx="28" cy="5" rx="12" ry="15" fill="#202038"/>
        </g>
      </g>
    </svg>
  );
};

// Animated version
interface AnimatedCelShadedBoneCharacterProps {
  poseSequence: {pose: keyof typeof POSES; duration: number}[];
  x?: number;
  y?: number;
  scale?: number;
}

export const AnimatedCelShadedBoneCharacter: React.FC<AnimatedCelShadedBoneCharacterProps> = ({
  poseSequence,
  x = 960,
  y = 540,
  scale = 1,
}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();

  let currentFrame = 0;
  const ranges: {pose: Pose; startFrame: number; endFrame: number}[] = [];

  for (const item of poseSequence) {
    const durationFrames = item.duration * fps;
    ranges.push({
      pose: POSES[item.pose],
      startFrame: currentFrame,
      endFrame: currentFrame + durationFrames,
    });
    currentFrame += durationFrames;
  }

  let currentPose = POSES.standing;

  for (let i = 0; i < ranges.length; i++) {
    const r = ranges[i];
    if (frame >= r.startFrame && frame < r.endFrame) {
      const nextPose = i < ranges.length - 1 ? ranges[i + 1].pose : r.pose;
      const transitionFrames = 12;
      const transitionStart = r.endFrame - transitionFrames;

      if (frame >= transitionStart) {
        const t = (frame - transitionStart) / transitionFrames;
        const eased = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
        currentPose = interpolatePose(r.pose, nextPose, eased);
      } else {
        currentPose = r.pose;
      }
      break;
    }
  }

  // Subtle breathing animation
  const bob = Math.sin(frame * 0.08) * 2;

  return <CelShadedBoneCharacter pose={currentPose} x={x} y={y + bob} scale={scale}/>;
};
