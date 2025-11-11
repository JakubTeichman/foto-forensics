import React, { useState, useRef, useLayoutEffect, useEffect } from "react";
import { Link, useLocation } from "react-router-dom";
import { Menu, X } from "lucide-react";
import { gsap } from "gsap";

const Header: React.FC = () => {
  const location = useLocation();
  const [isOpen, setIsOpen] = useState(false);
  const [maxHeight, setMaxHeight] = useState<number | null>(null);
  const navRef = useRef<HTMLDivElement | null>(null);
  const tlRef = useRef<gsap.core.Timeline | null>(null);

  const toggleMenu = () => {
    const tl = tlRef.current;
    if (!tl) return;

    if (!isOpen) {
      setIsOpen(true);
      tl.play(0);
    } else {
      tl.reverse();
      tl.eventCallback("onReverseComplete", () => setIsOpen(false));
    }
  };

  // Animacja GSAP
  useLayoutEffect(() => {
    const nav = navRef.current;
    if (!nav) return;

    const cards = nav.querySelectorAll(".menu-section");

    gsap.set(nav, { height: 64, overflow: "hidden" });
    gsap.set(cards, { y: 40, opacity: 0 });

    const tl = gsap.timeline({ paused: true });
    tl.to(nav, { height: "auto", duration: 0.4, ease: "power3.out" });
    tl.to(
      cards,
      { y: 0, opacity: 1, duration: 0.4, ease: "power3.out", stagger: 0.08 },
      "-=0.1"
    );

    tlRef.current = tl;
    return () => {
      tl.kill();
      tlRef.current = null;
    };
  }, []);

  // Ustal maksymalną wysokość sekcji, żeby wszystkie były równe
  useEffect(() => {
    const sections = document.querySelectorAll(".menu-section");
    if (sections.length > 0) {
      const max = Math.max(...Array.from(sections).map((s) => s.clientHeight)); 
      setMaxHeight(max);
    }
  }, [isOpen]);

  const isActive = (path: string) => location.pathname === path;

  return (
    <header className="fixed left-1/2 -translate-x-1/2 top-3 w-[90%] max-w-[900px] z-50">
      <div
        ref={navRef}
        className="relative rounded-[3rem] backdrop-blur-2xl bg-white/15 border border-white/20 shadow-[0_0_35px_rgba(0,255,180,0.25)] transition-all duration-500 overflow-hidden"
      >
        {/* Górny pasek */}
        <div className="flex items-center justify-between h-[64px] px-4 relative z-10">
          <Link
            to="/"
            className="absolute left-1/2 -translate-x-1/2 font-extrabold text-transparent text-lg md:text-xl bg-gradient-to-r from-[#00ffd0] to-[#00ff99] bg-clip-text tracking-widest"
          >
            FOTOFORENSICS
          </Link>

          <button
            onClick={toggleMenu}
            className="ml-auto text-[#00ffd0] hover:text-[#00ff99] transition-colors"
            aria-label="Toggle Menu"
          >
            {isOpen ? <X size={26} /> : <Menu size={26} />}
          </button>
        </div>

        {/* Dropdown sekcje */}
        <div
          className={`transition-all duration-300 ${
            isOpen ? "visible opacity-100" : "invisible opacity-0"
          }`}
        >
          <div
            className="
              flex flex-col md:flex-row md:justify-between md:items-stretch
              gap-3 md:gap-4 px-4 pb-5 md:pb-7
            "
          >
            {/* ABOUT */}
            <div
              className="menu-section flex-1 bg-white/10 backdrop-blur-lg border border-white/20 rounded-2xl p-5 shadow-inner flex flex-col justify-between hover:bg-white/15 transition-colors"
              style={{
                height: maxHeight ? `${maxHeight}px` : "auto",
              }}
            >
              <h3 className="text-transparent bg-clip-text bg-gradient-to-r from-[#00ffd0] to-[#00ff99] font-extrabold text-lg tracking-widest uppercase text-center md:text-left drop-shadow-[0_0_10px_rgba(0,255,200,0.3)] mb-2">
                ABOUT
              </h3>
              <Link
                to="/about"
                onClick={toggleMenu}
                className={`block text-gray-200 hover:text-[#00ffb0] transition-colors text-center md:text-left ${
                  isActive("/about") ? "text-[#00ffb0]" : ""
                }`}
              >
                About Project
              </Link>
            </div>

            {/* MODULES */}
            <div
              className="menu-section flex-1 bg-white/10 backdrop-blur-lg border border-white/20 rounded-2xl p-5 shadow-inner flex flex-col justify-between hover:bg-white/15 transition-colors"
              style={{
                height: maxHeight ? `${maxHeight}px` : "auto",
              }}
            >
              <h3 className="text-transparent bg-clip-text bg-gradient-to-r from-[#00ffd0] to-[#00ff99] font-extrabold text-lg tracking-widest uppercase text-center md:text-left drop-shadow-[0_0_10px_rgba(0,255,200,0.3)] mb-2">
                MODULES
              </h3>
              <div className="flex flex-col space-y-1 text-center md:text-left">
                <Link
                  to="/compare"
                  onClick={toggleMenu}
                  className={`hover:text-[#00ffb0] ${
                    isActive("/compare") ? "text-[#00ffb0]" : "text-gray-200"
                  }`}
                >
                  Compare
                </Link>
                <Link
                  to="/analysis"
                  onClick={toggleMenu}
                  className={`hover:text-[#00ffb0] ${
                    isActive("/analysis") ? "text-[#00ffb0]" : "text-gray-200"
                  }`}
                >
                  Analysis
                </Link>
                <Link
                  to="/stegano-compare"
                  onClick={toggleMenu}
                  className={`hover:text-[#00ffb0] ${
                    isActive("/stegano-compare")
                      ? "text-[#00ffb0]"
                      : "text-gray-200"
                  }`}
                >
                  Stegano + Integrity
                </Link>
              </div>
            </div>

            {/* CONTACT */}
            <div
              className="menu-section flex-1 bg-white/10 backdrop-blur-lg border border-white/20 rounded-2xl p-5 shadow-inner flex flex-col justify-between hover:bg-white/15 transition-colors"
              style={{
                height: maxHeight ? `${maxHeight}px` : "auto",
              }}
            >
              <h3 className="text-transparent bg-clip-text bg-gradient-to-r from-[#00ffd0] to-[#00ff99] font-extrabold text-lg tracking-widest uppercase text-center md:text-left drop-shadow-[0_0_10px_rgba(0,255,200,0.3)] mb-2">
                CONTACT
              </h3>
              <div className="flex flex-col space-y-1 text-center md:text-left">
                <Link
                  to="/add-reference"
                  onClick={toggleMenu}
                  className={`hover:text-[#00ffb0] ${
                    isActive("/add-reference")
                      ? "text-[#00ffb0]"
                      : "text-gray-200"
                  }`}
                >
                  Add Reference
                </Link>
                <Link
                  to="/contact"
                  onClick={toggleMenu}
                  className={`hover:text-[#00ffb0] ${
                    isActive("/contact") ? "text-[#00ffb0]" : "text-gray-200"
                  }`}
                >
                  Contact
                </Link>
              </div>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
