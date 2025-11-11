import React, { useRef, useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Renderer, Program, Mesh, Triangle, Vec2 } from "ogl";

// ======== 💡 Spotlight Card ========
interface Position {
  x: number;
  y: number;
}

interface SpotlightCardProps extends React.PropsWithChildren {
  className?: string;
  spotlightColor?: string;
}

const SpotlightCard: React.FC<SpotlightCardProps> = ({
  children,
  className = "",
  spotlightColor = "rgba(255, 255, 255, 0.25)",
}) => {
  const divRef = useRef<HTMLDivElement>(null);
  const [isFocused, setIsFocused] = useState(false);
  const [position, setPosition] = useState<Position>({ x: 0, y: 0 });
  const [opacity, setOpacity] = useState(0);

  const handleMouseMove: React.MouseEventHandler<HTMLDivElement> = (e) => {
    if (!divRef.current || isFocused) return;
    const rect = divRef.current.getBoundingClientRect();
    setPosition({ x: e.clientX - rect.left, y: e.clientY - rect.top });
  };

  const handleFocus = () => {
    setIsFocused(true);
    setOpacity(0.6);
  };
  const handleBlur = () => {
    setIsFocused(false);
    setOpacity(0);
  };
  const handleMouseEnter = () => setOpacity(0.6);
  const handleMouseLeave = () => setOpacity(0);

  return (
    <div
      ref={divRef}
      onMouseMove={handleMouseMove}
      onFocus={handleFocus}
      onBlur={handleBlur}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      className={`relative rounded-3xl border border-neutral-800 bg-neutral-900 overflow-hidden p-8 transition-all duration-300 ${className}`}
    >
      <div
        className="pointer-events-none absolute inset-0 transition-opacity duration-500 ease-in-out"
        style={{
          opacity,
          background: `radial-gradient(circle at ${position.x}px ${position.y}px, ${spotlightColor}, transparent 80%)`,
        }}
      />
      {children}
    </div>
  );
};

// ======== 🌌 Background Shader ========
const useShaderBackground = () => {
  useEffect(() => {
    const canvas = document.createElement("canvas");
    canvas.className = "absolute inset-0 w-full h-full";
    canvas.style.zIndex = "-1";
    document.body.appendChild(canvas);

    const renderer = new Renderer({ canvas, dpr: 2 });
    const gl = renderer.gl;
    const program = new Program(gl, {
      vertex: `
        attribute vec2 position;
        void main() {
          gl_Position = vec4(position, 0.0, 1.0);
        }
      `,
      fragment: `
        precision lowp float;
        uniform vec2 uResolution;
        uniform float uTime;

        float random(vec2 st) {
          return fract(sin(dot(st.xy, vec2(12.9898,78.233))) * 43758.5453);
        }

        void main() {
          vec2 st = gl_FragCoord.xy / uResolution.xy;
          float color = 0.0;
          color += sin(st.x * 10.0 + uTime * 0.2) * 0.1;
          color += random(st) * 0.02;
          gl_FragColor = vec4(vec3(0.0, color, color * 2.0), 1.0);
        }
      `,
      uniforms: {
        uTime: { value: 0 },
        uResolution: { value: new Vec2() },
      },
    });

    const mesh = new Mesh(gl, { geometry: new Triangle(gl), program });

    let time = 0;
    const resize = () => {
      renderer.setSize(window.innerWidth, window.innerHeight);
      program.uniforms.uResolution.value.set(window.innerWidth, window.innerHeight);
    };
    window.addEventListener("resize", resize);
    resize();

    const update = () => {
      time += 0.05;
      program.uniforms.uTime.value = time;
      renderer.render({ scene: mesh });
      requestAnimationFrame(update);
    };
    update();

    return () => {
      window.removeEventListener("resize", resize);
      canvas.remove();
    };
  }, []);
};

// ======== 🌙 MAIN ABOUT PAGE ========
interface AboutProps {
  setActiveTab: (tab: string) => void;
}

const About: React.FC<AboutProps> = ({ setActiveTab }) => {
  const navigate = useNavigate();
  useShaderBackground();

  const handleStartAnalysis = () => {
    setActiveTab("analysis");
    navigate("/analysis");
  };

  const modules = [
    {
      icon: "fas fa-project-diagram",
      title: "Image Comparison",
      description:
        "Compare an evidence image with reference photos to confirm if they come from the same device.\nUse reference images with identical parameters for best accuracy.\n\nDenoising methods:\n• BM3D – more precise, slower\n• Wavelet – faster, slightly less accurate.",
      link: "/compare",
    },
    {
      icon: "fas fa-microscope",
      title: "Image Analysis",
      description:
        "Explore binary and metadata structure.\nInspect hexadecimal and ASCII data, check EXIF and GPS info, visualize locations, and detect steganography.",
      link: "/analysis",
    },
    {
      icon: "fas fa-link",
      title: "Steganography + Integrity",
      description:
        "Compare two images to verify integrity and reveal hidden data or possible tampering.",
      link: "/stegano-compare",
    },
  ];

  const cards = [
    {
      icon: "fas fa-search",
      title: "Detect Manipulation",
      description:
        "Inspect hidden data in image and compare them to verify integrity.",
    },
    {
      icon: "fas fa-code",
      title: "Metadata Analysis",
      description:
        "Extract EXIF, GPS data, HEX view and steganography raport.",
    },
    {
      icon: "fas fa-shield-alt",
      title: "Forensic Reports",
      description:
        "Generate structured forensic raport for personal use.",
    },
  ];

  return (
    <div className="relative flex flex-col items-center overflow-hidden min-h-screen pt-24 pb-24 text-white bg-black">
      {/* CONTENT */}
      <div className="relative z-10 w-full flex flex-col items-center">
        {/* Header */}
        <div className="relative max-w-4xl text-center mb-16">
          <h2 className="text-5xl font-bold mb-6 text-teal-400 leading-tight">
            Free Photo Forensics Analysis
          </h2>
          <p className="text-lg text-gray-300 mb-8">
            Detect manipulations and hidden data, check photo source, analyze metadata,
            check photo location, verify integrity.
          </p>
          <button
            onClick={handleStartAnalysis}
            className="bg-gradient-to-r from-teal-500 to-green-400 text-black font-bold px-8 py-3 rounded-full hover:from-teal-600 hover:to-green-500 transition-all"
          >
            Start Analysis
          </button>
        </div>

        {/* Cards */}
        <div className="relative grid grid-cols-1 md:grid-cols-3 gap-8 w-full max-w-5xl mb-20">
          {cards.map((card, i) => (
            <SpotlightCard key={i}>
              <div className="text-teal-400 text-4xl mb-4">
                <i className={card.icon}></i>
              </div>
              <h3 className="text-xl font-bold mb-3 text-white">{card.title}</h3>
              <p className="text-gray-400">{card.description}</p>
            </SpotlightCard>
          ))}
        </div>

        {/* Modules */}
        <div className="relative flex flex-col space-y-8 w-full max-w-5xl px-4">
          {modules.map((mod, i) => (
            <SpotlightCard key={i} className="p-6 border border-teal-700 hover:border-teal-500 shadow-md hover:shadow-teal-600/20">
              <div className="flex flex-col md:flex-row items-center md:items-start">
                <div className="text-teal-400 text-4xl mb-4 md:mb-0 md:mr-6 shrink-0">
                  <i className={mod.icon}></i>
                </div>
                <div className="flex flex-col text-left md:text-justify">
                  <h3 className="text-2xl font-bold mb-3 text-white">{mod.title}</h3>
                  <p className="text-gray-400 mb-6 whitespace-pre-line leading-relaxed">
                    {mod.description}
                  </p>
                  <button
                    onClick={() => navigate(mod.link)}
                    className="self-start bg-gradient-to-r from-teal-500 to-green-400 text-black font-bold px-6 py-2 rounded-full hover:from-teal-600 hover:to-green-500 transition-all"
                  >
                    Go to Module
                  </button>
                </div>
              </div>
            </SpotlightCard>
          ))}
        </div>
      </div>
    </div>
  );
};

export default About;
