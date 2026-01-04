import {useCurrentFrame, useVideoConfig} from 'remotion';

interface Pose {
  rightShoulder: number;
  rightElbow: number;
  leftShoulder: number;
  leftElbow: number;
  rightHip: number;
  rightKnee: number;
  leftHip: number;
  leftKnee: number;
}

export const POSES: Record<string, Pose> = {
  standing: { rightShoulder: 15, rightElbow: 20, leftShoulder: 15, leftElbow: 20, rightHip: 5, rightKnee: 0, leftHip: 5, leftKnee: 0 },
  waving: { rightShoulder: 150, rightElbow: 35, leftShoulder: 15, leftElbow: 20, rightHip: 5, rightKnee: 0, leftHip: 5, leftKnee: 0 },
  pointing: { rightShoulder: 90, rightElbow: 0, leftShoulder: 15, leftElbow: 20, rightHip: 5, rightKnee: 0, leftHip: 5, leftKnee: 0 },
  presenting: { rightShoulder: 120, rightElbow: 10, leftShoulder: 15, leftElbow: 20, rightHip: 5, rightKnee: 0, leftHip: 5, leftKnee: 0 },
  thinking: { rightShoulder: 110, rightElbow: 140, leftShoulder: 15, leftElbow: 20, rightHip: 5, rightKnee: 0, leftHip: 5, leftKnee: 0 },
  celebrating: { rightShoulder: 160, rightElbow: 20, leftShoulder: 160, leftElbow: 20, rightHip: 5, rightKnee: 0, leftHip: 5, leftKnee: 0 },
  jumping: { rightShoulder: 140, rightElbow: 10, leftShoulder: 140, leftElbow: 10, rightHip: 25, rightKnee: 35, leftHip: 25, leftKnee: 35 },
  walking1: { rightShoulder: 30, rightElbow: 40, leftShoulder: 5, leftElbow: 10, rightHip: -10, rightKnee: 15, leftHip: 20, leftKnee: 30 },
  walking2: { rightShoulder: 5, rightElbow: 10, leftShoulder: 30, leftElbow: 40, rightHip: 20, rightKnee: 30, leftHip: -10, leftKnee: 15 },
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

// Calculate point at angle and distance from origin
function pt(x: number, y: number, angleDeg: number, dist: number) {
  const rad = angleDeg * Math.PI / 180;
  return { x: x + Math.sin(rad) * dist, y: y + Math.cos(rad) * dist };
}

// Get perpendicular offset at a point along a limb
function perpOffset(start: {x: number, y: number}, end: {x: number, y: number}, width: number, side: 'left' | 'right') {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const len = Math.sqrt(dx * dx + dy * dy) || 1;
  const mult = side === 'left' ? 1 : -1;
  return {
    x: (-dy / len) * width / 2 * mult,
    y: (dx / len) * width / 2 * mult
  };
}

interface Props {
  pose: Pose;
  x?: number;
  y?: number;
  scale?: number;
}

export const GoldenCharacter: React.FC<Props> = ({ pose, x = 960, y = 540, scale = 1 }) => {
  // Match golden image dimensions
  const headR = 115;
  const neckW = 60;
  const shoulderW = 130; // Half-width from center to shoulder edge
  const bodyBottomW = 55;

  const upperArmW = 45;
  const lowerArmW = 40;
  const handR = 35;

  const upperLegW = 50;
  const lowerLegW = 45;
  const footR = 30;

  // Bone lengths
  const upperArmLen = 90;
  const lowerArmLen = 85;
  const upperLegLen = 110;
  const lowerLegLen = 100;

  // Key Y positions (relative to head center at 0)
  const headY = 0;
  const neckY = headR - 10;
  const shoulderY = neckY + 30;
  const bodyBottomY = shoulderY + 200;
  const hipY = bodyBottomY - 30;

  // Shoulder attachment points
  const shoulderX = 55;
  const hipX = 40;

  // Calculate bone joint positions
  // Right arm
  const rShoulder = { x: shoulderX, y: shoulderY };
  const rElbow = pt(rShoulder.x, rShoulder.y, pose.rightShoulder, upperArmLen);
  const rWrist = pt(rElbow.x, rElbow.y, pose.rightShoulder + pose.rightElbow, lowerArmLen);

  // Left arm (mirrored)
  const lShoulder = { x: -shoulderX, y: shoulderY };
  const lElbow = pt(lShoulder.x, lShoulder.y, -pose.leftShoulder, upperArmLen);
  const lWrist = pt(lElbow.x, lElbow.y, -pose.leftShoulder - pose.leftElbow, lowerArmLen);

  // Right leg
  const rHip = { x: hipX, y: hipY };
  const rKnee = pt(rHip.x, rHip.y, pose.rightHip, upperLegLen);
  const rAnkle = pt(rKnee.x, rKnee.y, pose.rightHip - pose.rightKnee, lowerLegLen);

  // Left leg (mirrored)
  const lHip = { x: -hipX, y: hipY };
  const lKnee = pt(lHip.x, lHip.y, -pose.leftHip, upperLegLen);
  const lAnkle = pt(lKnee.x, lKnee.y, -pose.leftHip + pose.leftKnee, lowerLegLen);

  // Generate the unified body silhouette path
  // We trace: neck -> right shoulder -> right arm -> right hand (around) ->
  //           back up arm -> body right side -> right leg -> right foot (around) ->
  //           back up leg -> body bottom -> left leg -> left foot -> back up ->
  //           body left side -> left arm -> left hand -> back up -> left shoulder -> neck

  const path: string[] = [];

  // === START: Top of neck (left side) ===
  path.push(`M ${-neckW/2} ${neckY}`);

  // Curve to left shoulder
  path.push(`Q ${-shoulderW * 0.3} ${shoulderY - 20} ${-shoulderW * 0.8} ${shoulderY}`);

  // === LEFT ARM (going down the outside, around hand, back up inside) ===
  // Shoulder to elbow - outer edge
  const lUpperOut = perpOffset(lShoulder, lElbow, upperArmW, 'left');
  path.push(`C ${lShoulder.x + lUpperOut.x - 20} ${lShoulder.y + lUpperOut.y + 30} ${lElbow.x + lUpperOut.x} ${lElbow.y + lUpperOut.y - 30} ${lElbow.x + lUpperOut.x} ${lElbow.y + lUpperOut.y}`);

  // Elbow to wrist - outer edge
  const lLowerOut = perpOffset(lElbow, lWrist, lowerArmW, 'left');
  path.push(`C ${lElbow.x + lLowerOut.x} ${lElbow.y + lLowerOut.y + 20} ${lWrist.x + lLowerOut.x} ${lWrist.y + lLowerOut.y - 20} ${lWrist.x + lLowerOut.x} ${lWrist.y + lLowerOut.y}`);

  // Around the hand (semicircle)
  path.push(`A ${handR} ${handR} 0 1 0 ${lWrist.x - lLowerOut.x} ${lWrist.y - lLowerOut.y}`);

  // Wrist back to elbow - inner edge
  const lLowerIn = perpOffset(lElbow, lWrist, lowerArmW, 'right');
  path.push(`C ${lWrist.x + lLowerIn.x} ${lWrist.y + lLowerIn.y - 20} ${lElbow.x + lLowerIn.x} ${lElbow.y + lLowerIn.y + 20} ${lElbow.x + lLowerIn.x} ${lElbow.y + lLowerIn.y}`);

  // Elbow back to body - inner edge
  const lUpperIn = perpOffset(lShoulder, lElbow, upperArmW, 'right');
  path.push(`C ${lElbow.x + lUpperIn.x} ${lElbow.y + lUpperIn.y - 30} ${lShoulder.x + lUpperIn.x - 10} ${lShoulder.y + lUpperIn.y + 30} ${-bodyBottomW * 0.8} ${shoulderY + 60}`);

  // === LEFT SIDE OF BODY (going down) ===
  path.push(`C ${-bodyBottomW * 0.9} ${hipY - 80} ${-bodyBottomW * 0.7} ${hipY - 20} ${lHip.x - upperLegW/2} ${lHip.y}`);

  // === LEFT LEG (going down outside, around foot, back up inside) ===
  // Hip to knee - outer edge
  const lLegUpperOut = perpOffset(lHip, lKnee, upperLegW, 'left');
  path.push(`C ${lHip.x + lLegUpperOut.x} ${lHip.y + lLegUpperOut.y + 30} ${lKnee.x + lLegUpperOut.x} ${lKnee.y + lLegUpperOut.y - 20} ${lKnee.x + lLegUpperOut.x} ${lKnee.y + lLegUpperOut.y}`);

  // Knee to ankle - outer edge
  const lLegLowerOut = perpOffset(lKnee, lAnkle, lowerLegW, 'left');
  path.push(`C ${lKnee.x + lLegLowerOut.x} ${lKnee.y + lLegLowerOut.y + 20} ${lAnkle.x + lLegLowerOut.x} ${lAnkle.y + lLegLowerOut.y - 15} ${lAnkle.x + lLegLowerOut.x} ${lAnkle.y + lLegLowerOut.y}`);

  // Around the foot
  path.push(`A ${footR} ${footR} 0 1 0 ${lAnkle.x - lLegLowerOut.x} ${lAnkle.y - lLegLowerOut.y}`);

  // Ankle back to knee - inner edge
  const lLegLowerIn = perpOffset(lKnee, lAnkle, lowerLegW, 'right');
  path.push(`C ${lAnkle.x + lLegLowerIn.x} ${lAnkle.y + lLegLowerIn.y - 15} ${lKnee.x + lLegLowerIn.x} ${lKnee.y + lLegLowerIn.y + 20} ${lKnee.x + lLegLowerIn.x} ${lKnee.y + lLegLowerIn.y}`);

  // Knee back to hip - inner edge
  const lLegUpperIn = perpOffset(lHip, lKnee, upperLegW, 'right');
  path.push(`C ${lKnee.x + lLegUpperIn.x} ${lKnee.y + lLegUpperIn.y - 20} ${lHip.x + lLegUpperIn.x} ${lHip.y + lLegUpperIn.y + 30} ${lHip.x + upperLegW/2} ${lHip.y}`);

  // === BOTTOM OF BODY (crotch area) ===
  path.push(`Q ${0} ${bodyBottomY + 30} ${rHip.x - upperLegW/2} ${rHip.y}`);

  // === RIGHT LEG (going down outside, around foot, back up inside) ===
  // Hip to knee - outer edge (note: 'right' side is outer for right leg)
  const rLegUpperOut = perpOffset(rHip, rKnee, upperLegW, 'right');
  path.push(`C ${rHip.x + rLegUpperOut.x} ${rHip.y + rLegUpperOut.y + 30} ${rKnee.x + rLegUpperOut.x} ${rKnee.y + rLegUpperOut.y - 20} ${rKnee.x + rLegUpperOut.x} ${rKnee.y + rLegUpperOut.y}`);

  // Knee to ankle - outer edge
  const rLegLowerOut = perpOffset(rKnee, rAnkle, lowerLegW, 'right');
  path.push(`C ${rKnee.x + rLegLowerOut.x} ${rKnee.y + rLegLowerOut.y + 20} ${rAnkle.x + rLegLowerOut.x} ${rAnkle.y + rLegLowerOut.y - 15} ${rAnkle.x + rLegLowerOut.x} ${rAnkle.y + rLegLowerOut.y}`);

  // Around the foot
  path.push(`A ${footR} ${footR} 0 1 0 ${rAnkle.x - rLegLowerOut.x} ${rAnkle.y - rLegLowerOut.y}`);

  // Ankle back to knee - inner edge
  const rLegLowerIn = perpOffset(rKnee, rAnkle, lowerLegW, 'left');
  path.push(`C ${rAnkle.x + rLegLowerIn.x} ${rAnkle.y + rLegLowerIn.y - 15} ${rKnee.x + rLegLowerIn.x} ${rKnee.y + rLegLowerIn.y + 20} ${rKnee.x + rLegLowerIn.x} ${rKnee.y + rLegLowerIn.y}`);

  // Knee back to hip - inner edge
  const rLegUpperIn = perpOffset(rHip, rKnee, upperLegW, 'left');
  path.push(`C ${rKnee.x + rLegUpperIn.x} ${rKnee.y + rLegUpperIn.y - 20} ${rHip.x + rLegUpperIn.x} ${rHip.y + rLegUpperIn.y + 30} ${rHip.x + upperLegW/2} ${rHip.y}`);

  // === RIGHT SIDE OF BODY (going up) ===
  path.push(`C ${bodyBottomW * 0.7} ${hipY - 20} ${bodyBottomW * 0.9} ${hipY - 80} ${bodyBottomW * 0.8} ${shoulderY + 60}`);

  // === RIGHT ARM (going down outside, around hand, back up inside) ===
  // Body to shoulder/elbow - inner edge first
  const rUpperIn = perpOffset(rShoulder, rElbow, upperArmW, 'left');
  path.push(`C ${rShoulder.x + rUpperIn.x + 10} ${rShoulder.y + rUpperIn.y + 30} ${rElbow.x + rUpperIn.x} ${rElbow.y + rUpperIn.y - 30} ${rElbow.x + rUpperIn.x} ${rElbow.y + rUpperIn.y}`);

  // Elbow to wrist - inner edge
  const rLowerIn = perpOffset(rElbow, rWrist, lowerArmW, 'left');
  path.push(`C ${rElbow.x + rLowerIn.x} ${rElbow.y + rLowerIn.y + 20} ${rWrist.x + rLowerIn.x} ${rWrist.y + rLowerIn.y - 20} ${rWrist.x + rLowerIn.x} ${rWrist.y + rLowerIn.y}`);

  // Around the hand
  path.push(`A ${handR} ${handR} 0 1 1 ${rWrist.x - rLowerIn.x} ${rWrist.y - rLowerIn.y}`);

  // Wrist back to elbow - outer edge
  const rLowerOut = perpOffset(rElbow, rWrist, lowerArmW, 'right');
  path.push(`C ${rWrist.x + rLowerOut.x} ${rWrist.y + rLowerOut.y - 20} ${rElbow.x + rLowerOut.x} ${rElbow.y + rLowerOut.y + 20} ${rElbow.x + rLowerOut.x} ${rElbow.y + rLowerOut.y}`);

  // Elbow back to shoulder - outer edge
  const rUpperOut = perpOffset(rShoulder, rElbow, upperArmW, 'right');
  path.push(`C ${rElbow.x + rUpperOut.x} ${rElbow.y + rUpperOut.y - 30} ${rShoulder.x + rUpperOut.x + 20} ${rShoulder.y + rUpperOut.y + 30} ${shoulderW * 0.8} ${shoulderY}`);

  // Right shoulder back to neck
  path.push(`Q ${shoulderW * 0.3} ${shoulderY - 20} ${neckW/2} ${neckY}`);

  // Close the neck
  path.push('Z');

  const bodyPath = path.join(' ');

  return (
    <svg width={1920} height={1080} style={{ position: 'absolute', left: 0, top: 0 }}>
      <defs>
        <radialGradient id="bodyFill" cx="50%" cy="25%" r="75%">
          <stop offset="0%" stopColor="#FFFFFF"/>
          <stop offset="50%" stopColor="#F5F5FA"/>
          <stop offset="100%" stopColor="#E8E8F0"/>
        </radialGradient>

        <filter id="dropShadow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur in="SourceAlpha" stdDeviation="9"/>
          <feOffset dx="6" dy="10" result="shadow"/>
          <feFlood floodColor="#1a1a2e" floodOpacity="0.35"/>
          <feComposite in2="shadow" operator="in"/>
          <feMerge>
            <feMergeNode/>
            <feMergeNode in="SourceGraphic"/>
          </feMerge>
        </filter>

        <clipPath id="headClip">
          <circle cx={0} cy={0} r={headR}/>
        </clipPath>
      </defs>

      <g transform={`translate(${x}, ${y}) scale(${scale})`}>
        {/* Ground Shadow */}
        <ellipse cx={0} cy={hipY + upperLegLen + lowerLegLen + 80} rx={180} ry={25} fill="rgba(30, 30, 50, 0.5)"/>

        <g filter="url(#dropShadow)">
          {/* Unified body silhouette */}
          <path
            d={bodyPath}
            fill="url(#bodyFill)"
            stroke="#202038"
            strokeWidth={12}
            strokeLinejoin="round"
            strokeLinecap="round"
          />

          {/* Head */}
          <circle cx={0} cy={headY} r={headR} fill="url(#bodyFill)" stroke="#202038" strokeWidth={12}/>

          {/* Head reflection */}
          <path
            d={`M ${headR * 0.2} ${-headR * 0.75} A ${headR * 0.74} ${headR * 0.74} 0 0 1 ${headR * 0.68} ${-headR * 0.35}`}
            stroke="#D0D0D8"
            strokeWidth={16}
            fill="none"
            strokeLinecap="round"
            opacity={0.5}
            clipPath="url(#headClip)"
          />

          {/* Eyes */}
          <ellipse cx={-27} cy={headY + 5} rx={13} ry={16} fill="#202038"/>
          <ellipse cx={27} cy={headY + 5} rx={13} ry={16} fill="#202038"/>

          {/* Eye highlights */}
          <circle cx={-30} cy={headY} r={4} fill="#FFFFFF" opacity={0.95}/>
          <circle cx={24} cy={headY} r={4} fill="#FFFFFF" opacity={0.95}/>
        </g>
      </g>
    </svg>
  );
};

export const AnimatedGoldenCharacter: React.FC<{
  poseSequence: {pose: keyof typeof POSES; duration: number}[];
  x?: number;
  y?: number;
  scale?: number;
}> = ({ poseSequence, x = 960, y = 540, scale = 1 }) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();

  let currentFrame = 0;
  const ranges: {pose: Pose; startFrame: number; endFrame: number}[] = [];
  for (const item of poseSequence) {
    const dur = item.duration * fps;
    ranges.push({ pose: POSES[item.pose], startFrame: currentFrame, endFrame: currentFrame + dur });
    currentFrame += dur;
  }

  let currentPose = POSES.standing;
  for (let i = 0; i < ranges.length; i++) {
    const r = ranges[i];
    if (frame >= r.startFrame && frame < r.endFrame) {
      const next = i < ranges.length - 1 ? ranges[i + 1].pose : r.pose;
      const transDur = 15; // Smooth 0.5s transitions
      const transStart = r.endFrame - transDur;
      if (frame >= transStart) {
        const t = (frame - transStart) / transDur;
        const eased = t < 0.5 ? 2*t*t : 1 - Math.pow(-2*t + 2, 2) / 2;
        currentPose = interpolatePose(r.pose, next, eased);
      } else {
        currentPose = r.pose;
      }
      break;
    }
  }

  return <GoldenCharacter pose={currentPose} x={x} y={y + Math.sin(frame * 0.08) * 3} scale={scale}/>;
};
