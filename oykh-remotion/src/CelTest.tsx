import {AbsoluteFill} from 'remotion';
import {AnimatedCelShadedCharacter} from './CelShadedCharacter';

export const CelTest: React.FC = () => {
  return (
    <AbsoluteFill style={{
      background: 'linear-gradient(135deg, #45b8d1 0%, #6a6acd 30%, #c080a8 60%, #e8a060 100%)'
    }}>
      <AnimatedCelShadedCharacter
        poseSequence={[
          {pose: 'standing', duration: 1.2},
          {pose: 'waving', duration: 1.2},
          {pose: 'pointing', duration: 1.2},
          {pose: 'presenting', duration: 1.2},
          {pose: 'thinking', duration: 1.2},
          {pose: 'celebrating', duration: 1.2},
          {pose: 'jumping', duration: 1},
          {pose: 'walking1', duration: 0.4},
          {pose: 'walking2', duration: 0.4},
          {pose: 'walking1', duration: 0.4},
          {pose: 'walking2', duration: 0.4},
          {pose: 'standing', duration: 1},
        ]}
        y={480}
        scale={0.85}
      />
    </AbsoluteFill>
  );
};
