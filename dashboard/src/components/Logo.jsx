import logoSrc from "./PawlPatrolLogo.png";

// The PNG has wide transparent margins around the shield; this is the
// shield's bounding box within it, so the logo can be sized by the shield
// itself rather than by the whole image.
const IMAGE = { w: 784, h: 653 };
const SHIELD = { x: 141, y: 55, w: 507, h: 545 };

export default function Logo({ height, className = "" }) {
  return (
    <div
      className={`relative overflow-hidden shrink-0 ${className}`}
      style={{ height, aspectRatio: `${SHIELD.w} / ${SHIELD.h}` }}
    >
      <img
        src={logoSrc}
        alt="Pawl Patrol"
        draggable={false}
        style={{
          position: "absolute",
          maxWidth: "none",
          width: `${(IMAGE.w / SHIELD.w) * 100}%`,
          left: `${(-SHIELD.x / SHIELD.w) * 100}%`,
          top: `${(-SHIELD.y / SHIELD.h) * 100}%`,
        }}
      />
    </div>
  );
}
