"""Built-in tool policy: which of Google Search / URL context / Maps is on for
this request, and how they combine with function tools per model generation.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class _Tools:
    _PATTERN_URL = re.compile(r"https?://[^\s<>\"')\]]+")

    @staticmethod
    def _parse_lat_lng(*parts) -> Optional[Tuple[float, float]]:
        """(lat, lng) from a (lat, lng) pair or a "lat, lng" string; None when
        missing or unparsable, so callers fall through to the next source."""
        if len(parts) == 2:
            pairs = [(str(parts[0]), str(parts[1]))]
        elif len(parts) == 1:
            pieces = [p.strip() for p in str(parts[0] or "").split(",")]
            pairs = [(pieces[0], pieces[1])] if len(pieces) == 2 else []
        else:
            pairs = []
        for lat, lng in pairs:
            try:
                return (float(lat), float(lng))
            except ValueError:
                continue
        return None

    def _last_user_url(self, text: str) -> Optional[str]:
        """The first non-YouTube URL in the user's last message, if any."""
        for match in self._PATTERN_URL.finditer(text or ""):
            host = urlparse(match.group(0)).netloc.lower()
            if host == "youtube.com" or host.endswith(".youtube.com") or host == "youtu.be" or host.endswith(".youtu.be"):
                continue
            return match.group(0)
        return None

    def _resolve_builtin_flags(
        self, metadata: dict, valves, user_valves, info: dict, last_user_text: str
    ) -> dict:
        """Which built-in tools are on for this request, plus the Maps location."""
        metadata = metadata or {}
        search = bool(
            (
                valves.WEB_SEARCH_DEFAULT
                or metadata.get("gemini_web_search")
                or metadata.get("web_search_enforced")
            )
            and info.get("search")
        )

        mode = valves.URL_CONTEXT or "auto"
        url_context_auto = False
        if mode == "off":
            url_context = False
        elif mode == "always":
            url_context = True
        elif mode == "with_search":
            url_context = search
        else:
            url = self._last_user_url(last_user_text)
            url_context = bool(search or url)
            url_context_auto = bool(url and not search)
        url_context = url_context and bool(info.get("url_context"))

        maps = bool(
            (valves.MAPS_DEFAULT or metadata.get("gemini_maps")) and info.get("maps")
        )

        lat_lng = None
        if user_valves is not None:
            lat_lng = self._parse_lat_lng(
                user_valves.MAPS_LATITUDE, user_valves.MAPS_LONGITUDE
            )
        if lat_lng is None:
            raw = (metadata.get("variables") or {}).get("{{USER_LOCATION}}")
            if raw:
                lat_lng = self._parse_lat_lng(raw)
        if lat_lng is None:
            lat_lng = self._parse_lat_lng(valves.MAPS_LATITUDE, valves.MAPS_LONGITUDE)

        return {
            "search": search,
            "url_context": url_context,
            "url_context_auto": url_context_auto,
            "maps": maps,
            "lat_lng": lat_lng,
        }

    def _assemble_tools(
        self, info: dict, flags: dict, decls: List
    ) -> Tuple[List, Optional[object], List[str]]:
        """The Tool list for one call, its ToolConfig, and the degradation notes."""
        model = info.get("name") or "this model"
        notes: List[str] = []
        decls = list(decls or [])

        search = bool(flags.get("search"))
        maps = bool(flags.get("maps"))
        url_context = bool(flags.get("url_context"))
        url_context_auto = bool(flags.get("url_context_auto"))
        lat_lng = flags.get("lat_lng")

        if decls and not info.get("functions", True):
            notes.append(
                f"Function tools disabled for this message: {model} does not support function calling"
            )
            decls = []

        if search and maps and not info.get("maps_with_search"):
            search = False
            notes.append(
                f"Google Search disabled for this message: {model} cannot combine it with Google Maps"
            )

        builtins = search or url_context or maps
        if decls and builtins and not info.get("builtin_with_functions"):
            # The URL context that a URL in the user's message pulled in is the
            # softest dependency, so it goes first and silently.
            if url_context_auto:
                url_context = False
                builtins = search or maps
            if builtins:
                if self.valves.BUILTIN_TOOL_CONFLICT == "prefer_functions":
                    search = url_context = maps = False
                    builtins = False
                    notes.append(
                        f"Google Search/Maps disabled for this message: {model} cannot combine them with function tools"
                    )
                else:
                    decls = []
                    notes.append(
                        f"Function tools disabled for this message: {model} cannot combine them with Google Search/Maps"
                    )

        tools: List = []
        if decls:
            tools.append(gtypes.Tool(function_declarations=decls))
        if search:
            tools.append(gtypes.Tool(google_search=gtypes.GoogleSearch()))
        if url_context:
            tools.append(gtypes.Tool(url_context=gtypes.UrlContext()))
        if maps:
            tools.append(gtypes.Tool(google_maps=gtypes.GoogleMaps()))

        retrieval = None
        if maps and lat_lng:
            retrieval = gtypes.RetrievalConfig(
                lat_lng=gtypes.LatLng(latitude=lat_lng[0], longitude=lat_lng[1])
            )

        tool_config = None
        if decls and builtins:
            # Both sides stay: the SDK executes the built-ins server-side and
            # reports them back as regular tool calls.
            kwargs: dict = {}
            if GENAI_HAS_SERVER_SIDE_INVOCATIONS:
                kwargs["include_server_side_tool_invocations"] = True
            kwargs["function_calling_config"] = gtypes.FunctionCallingConfig(
                mode="VALIDATED"
            )
            if retrieval:
                kwargs["retrieval_config"] = retrieval
            tool_config = gtypes.ToolConfig(**kwargs)
        elif builtins and retrieval:
            tool_config = gtypes.ToolConfig(retrieval_config=retrieval)

        return tools, tool_config, notes

    @staticmethod
    def _sanitize_function_name(name: str) -> str:
        """A name Gemini accepts: [A-Za-z0-9_.:-], letter or _ first, at most 64 chars."""
        cleaned = re.sub(r"[^A-Za-z0-9_.:-]", "_", str(name or ""))
        if not cleaned or not (cleaned[0].isalpha() or cleaned[0] == "_"):
            cleaned = "_" + cleaned
        return cleaned[:64]

    def _build_function_declarations(self, ctx) -> list:
        """The FunctionDeclarations for this turn's executable tools.

        User tools first, then built-in tools: an original name offered twice
        is declared once, a sanitised-name collision gets a _2/_3/... suffix,
        and ``ctx.turn.name_map`` maps every declared name back to the
        callable's original name for execution and rendering.
        """
        decls: List[gtypes.FunctionDeclaration] = []
        offered_originals: set = set()
        taken: set = set()
        for source in (ctx.tools or {}, ctx.builtin_tools or {}):
            for original, entry in source.items():
                original = str(original)
                if original in offered_originals:
                    continue
                if original.startswith("_"):
                    continue
                if not isinstance(entry, dict) or not entry.get("callable"):
                    continue
                offered_originals.add(original)
                spec = entry.get("spec")
                spec = spec if isinstance(spec, dict) else {}
                candidate = self._sanitize_function_name(original)
                suffix = 2
                while candidate in taken:
                    candidate = f"{self._sanitize_function_name(original)}_{suffix}"
                    suffix += 1
                taken.add(candidate)
                ctx.turn.name_map[candidate] = original
                parameters = spec.get("parameters")
                if not isinstance(parameters, dict):
                    parameters = {"type": "object", "properties": {}}
                decls.append(
                    gtypes.FunctionDeclaration(
                        name=candidate,
                        description=spec.get("description") or "",
                        parameters_json_schema=parameters,
                    )
                )
        decls.sort(key=lambda d: d.name)
        ctx.turn.functions_offered = bool(decls)
        return decls
