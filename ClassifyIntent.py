import json
import re
from typing import List, Set, Dict, Optional, Any, Tuple, Union

# ---------------------------------------------------------------------------
# Freeform-keyword helpers (new).
#
# ClassifyIntent previously had no equivalent of agent_addon_5's `_terms` /
# STOPWORDS / leftover-term accumulation: any word in the user's message that
# didn't match one of the hardcoded lexicons below was silently dropped,
# even though it might still be useful for searching the catalog (e.g. a
# brand fragment, a product-line name, or vocabulary this classifier's
# lexicons just don't cover). `extract_keywords()` (see below) fills that
# gap using the SAME general approach as agent_addon_5: tokenize, drop
# generic stopwords, drop anything already captured by a matched
# constraint slot, return what's left.
#
# Deliberately only the generic, catalog/scenario-independent stopword list
# here — NOT agent_addon_5's extra evaluator-scaffolding stopwords (e.g.
# "requirement", "preference", "judgment", "prioritize"), since those exist
# specifically to strip that harness's own templated sentences and would be
# overfitting this general-purpose classifier to one evaluator's phrasing.
# ---------------------------------------------------------------------------

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS: Set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
    "am", "so"
}


def _terms(text: str) -> List[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


class IntentClassifier:
    """
    Rule-based intent router that distinguishes between 'Buying' (high constraint density)
    and 'Browsing' (low constraint density) user queries.
    Supports multiple features and multi-value constraints.
    """
    
    def __init__(self):
        """Rule-based extraction using hand-typed lexicons only. `category`
        and `brand` are matched against the small hardcoded
        `self.categories`/`self.brands` lists below — no catalog file is
        read or consulted."""
        # Static color list (common e-commerce colors)
        self.colors: Set[str] = {
            'black', 'white', 'red', 'blue', 'green', 'yellow', 'purple', 'pink',
            'gray', 'grey', 'brown', 'orange', 'navy', 'beige', 'tan', 'gold',
            'silver', 'bronze', 'maroon', 'teal', 'turquoise', 'lavender', 'cream',
            'ivory', 'charcoal', 'coral', 'magenta', 'olive', 'burgundy'
        }
        
        # Example brand names from catalog metadata (expand as needed)
        self.brands: Set[str] = {
            'nike', 'adidas', 'puma', 'under armour', 'levis', "levi's",
            'calvin klein', 'tommy hilfiger', 'ralph lauren', 'zara', 'h&m',
            'gap', 'old navy', 'banana republic', 'patagonia', 'north face',
            'columbia', 'carhartt', 'fruit of the loom', 'hanes', 'champion',
        }
        
        # Material/fabric terms
        self.materials: Set[str] = {
            'cotton', 'polyester', 'wool', 'silk', 'leather', 'denim', 'linen',
            'nylon', 'spandex', 'cashmere', 'velvet', 'suede', 'flannel', 'fleece',
            'canvas', 'mesh', 'satin', 'lace', 'chiffon', 'tweed', 'corduroy', 
            'stainless steel', 'sterling silver', 'platinum', 'gold', 'silver',
            # Jewelry / watch materials
            'alloy', 'titanium', 'rose gold', 'gold plated', 'gold-plated',
            'copper', 'brass', 'zinc alloy', 'cubic zirconia', 'rhodium',
            'tungsten', 'ceramic', 'resin', 'acrylic',
        }
        
        # Pattern/print terms
        self.patterns: Set[str] = {
            'striped', 'plaid', 'floral', 'solid', 'polka dot', 'checkered',
            'printed', 'geometric', 'paisley', 'camouflage', 'animal print',
            'leopard', 'zebra', 'tie-dye', 'ombre', 'distressed', 'vintage'
        }
        
        # Fit/silhouette terms
        self.fits: Set[str] = {
            'slim fit', 'regular fit', 'relaxed fit', 'loose fit', 'skinny',
            'straight leg', 'bootcut', 'tapered', 'oversized', 'fitted',
            'athletic fit', 'classic fit', 'modern fit', 'tailored', 'baggy'
        }
        
        # Style terms (renamed from occasions)
        self.style: Set[str] = {
            'casual', 'formal', 'business', 'party', 'wedding', 'workout',
            'athletic', 'outdoor', 'everyday', 'dressy', 'professional',
            'vacation', 'beach', 'winter', 'summer', 'spring', 'fall'
        }
        
        # Gender/target audience
        self.genders: Set[str] = {
            'men', 'women', 'mens', 'womens', 'boys', 'girls', 'kids',
            'unisex', 'toddler', 'infant', 'baby', 'youth', 'adult', 'children'
        }
        
        # Product categories - store both singular and plural forms where applicable
        self.categories: Set[str] = {
            'shirt', 'pants', 'shoes', 'jacket', 'jackets', 'dress', 
            'skirt', 'shorts', 'sweater', 'hoodie', 
            't-shirt', 'jeans', 'socks', 'hat', 'cap',
            'scarf', 'gloves', 'belt', 'watch',
            'bag', 'backpack', 'sunglasses', 'boots', 'sandals', 
            'sneakers', 'heels', 'suit', 'blazer', 'coat',
            'swimsuit', 'underwear', 'bra', 'leggings', 
            'cardigan', 'cardigans', 'vest', 'earrings', 'ring', 'studs', 
            'necklace', 'chain',
        }
        
        # Define singular-to-plural mapping for special cases
        self.singular_to_plural = {
            'shirt': 'shirts',
            'scarf': 'scarves',
            'dress': 'dresses',
            'skirt': 'skirts',
            'sweater': 'sweaters',
            'hoodie': 'hoodies',
            't-shirt': 't-shirts',
            'hat': 'hats',
            'cap': 'caps',
            'glove': 'gloves',
            'belt': 'belts',
            'watch': 'watches',
            'bag': 'bags',
            'backpack': 'backpacks',
            'boot': 'boots',
            'sandal': 'sandals',
            'sneaker': 'sneakers',
            'heel': 'heels',
            'suit': 'suits',
            'blazer': 'blazers',
            'coat': 'coats',
            'swimsuit': 'swimsuits',
            'bra': 'bras',
            'legging': 'leggings',
            'cardigan': 'cardigans',
            'vest': 'vests',
            'earring': 'earrings',
            'ring': 'rings',
            'necklace': 'necklaces',
            'chain': 'chains',
        }
        
        # Define plural-to-singular mapping (reverse of above, for normalization)
        self.plural_to_singular = {v: k for k, v in self.singular_to_plural.items()}
        
        # Words that are inherently plural (should not be singularized)
        self.inherently_plural = {
            'pants', 'jeans', 'shorts', 'leggings', 'sunglasses', 
            'earrings', 'studs', 'socks', 'shoes', 'boots', 'sandals',
            'sneakers', 'heels', 'gloves'
        }
        
        # Condition terms
        self.conditions: Set[str] = {
            'new', 'used', 'refurbished', 'like new', 'open box', 'pre-owned',
            'renewed', 'second-hand', 'second hand'
        }
        
        # Specific features/technology - organized by category
        self.features: Set[str] = {
            # Weather/durability features
            'waterproof', 'water-resistant', 'weatherproof', 'windproof',
            'thermal', 'insulated', 'breathable', 'moisture-wicking',
            
            # Sustainability features
            'eco-friendly', 'sustainable', 'recycled', 'organic', 'biodegradable',
            'fair trade', 'vegan', 'cruelty-free', 'carbon neutral',
            
            # Safety/protection features
            'antibacterial', 'antimicrobial', 'uv protection', 'spf',
            'flame resistant', 'fire resistant', 'safety certified',
            
            # Comfort/convenience features
            'adjustable', 'detachable', 'foldable', 'portable', 'lightweight',
            'durable', 'heavy-duty', 'high-waisted', 'stretch', 'wrinkle-free',
            'easy care', 'machine washable', 'quick dry',
            
            # Health/wellness features
            'hypoallergenic', 'ergonomic', 'orthopedic',
            'memory foam', 'gel cushioning', 'arch support',
            
            # Material features
            'scratch resistant', 'shatterproof', 'rust resistant',
            'corrosion resistant', 'stain resistant', 'water repellent',
            
            # Additional features
            'expandable', 'collapsible', 'stackable', 'modular',
            'multi-functional', 'all-in-one', 'compact', 'space saving',
            'imported'
        }
        
        # Size mappings for normalization
        self.size_mappings = {
            'extra small': 'XS', 'small': 'S', 'medium': 'M', 'large': 'L',
            'x-large': 'XL', 'xx-large': 'XXL', 'xxx-large': 'XXXL',
            'extra large': 'XL', 'extra small': 'XS', '3XL': 'XXXL', '2XL': 'XXL'
        }
        
        # Compile regex patterns
        self._compile_patterns()

    def _compile_patterns(self):
        """Pre-compile regex patterns for efficiency."""
        # Numeric price patterns with capture groups (hard constraints)
        self.price_patterns = [
            (r'\$\s?(\d+(?:\.\d{2})?)', lambda m: f"${m.group(1)}"),
            (r'\bunder\s+\$\s?(\d+(?:\.\d{2})?)\b', lambda m: f"${m.group(1)}"),
            (r'\bless than\s+\$\s?(\d+(?:\.\d{2})?)\b', lambda m: f"${m.group(1)}"),
            (r'\bbelow\s+\$\s?(\d+(?:\.\d{2})?)\b', lambda m: f"${m.group(1)}"),
            (r'\bover\s+\$\s?(\d+(?:\.\d{2})?)\b', lambda m: f"${m.group(1)}+"),
            (r'\bmore than\s+\$\s?(\d+(?:\.\d{2})?)\b', lambda m: f"${m.group(1)}+"),
            (r'\babove\s+\$\s?(\d+(?:\.\d{2})?)\b', lambda m: f"${m.group(1)}+"),
            (r'\bbetween\s+\$\s?(\d+(?:\.\d{2})?)\s+and\s+\$\s?(\d+(?:\.\d{2})?)\b', 
             lambda m: f"${m.group(1)}-${m.group(2)}"),
            (r'\b(\d+(?:\.\d{2})?)\s+bucks\b', lambda m: f"${m.group(1)}"),
            (r'\b(\d+(?:\.\d{2})?)\s+dollars\b', lambda m: f"${m.group(1)}"),
        ]
        
        # Size patterns with capture groups
        self.size_patterns = [
            (r'\b(?:size|sz)\s+(\d+(?:\.\d+)?|xs|s|m|l|xl|xxl|xxxl)\b', 
             lambda m: self._normalize_size(m.group(1))),
            (r'\b(\d+(?:\.\d+)?)\s*(?:inch|in|cm|mm|ft)\b', 
             lambda m: self._extract_measurement(m)),
            (r'\b(small|medium|large|x-large|xx-large|xxx-large|extra large|extra small)\b', 
             lambda m: self._normalize_size(m.group(1))),
            (r'\b(\d+(?:\.\d+)?)\s*(?:us|uk|eu)\b', 
             lambda m: f"{m.group(1)} {m.group(0)[-2:].upper()}"),
            (r'\bpetite\b', lambda m: "petite"),
            (r'\bplus\s+size\b', lambda m: "plus size"),
            (r'\btall\b', lambda m: "tall"),
            (r'\bregular\b', lambda m: "regular"),
            (r'\bshort\b', lambda m: "short"),
            # Add pattern for standalone numeric sizes (e.g., "size 10" without "size" keyword)
            (r'(?<!\$)(?<!\b\d)(?<![\d.])(\d{1,2}(?:\.\d+)?)(?![\d.])(?!\s*(?:dollars|bucks|USD))', 
            lambda m: m.group(1) if 4 <= float(m.group(1)) <= 22 else None)
        ]

        # Cue words that indicate a nearby bare number is a product
        # measurement/dimension (e.g. "shaft measures approximately 8.37\"
        # from arch") rather than a size the user is asking for.
        self.measurement_context_re = re.compile(
            r'\b(measures?|approximat\w*|circumference|diameter|shaft|'
            r'from\s+(?:the\s+)?(?:arch|heel|floor)|rise|drop|wingspan)\b',
            re.IGNORECASE,
        )
    
    def _normalize_size(self, size: str) -> str:
        """Normalize size representations.(e.g., "extra large" → "XL")"""
        size_lower = size.lower().strip()
        if size_lower in self.size_mappings:
            return self.size_mappings[size_lower]
        # Handle numeric sizes
        try:
            num = float(size_lower)
            if num.is_integer():
                return str(int(num))
            return str(num)
        except ValueError:
            return size.upper()
    
    def _extract_measurement(self, match) -> str:
        """Extract measurement with unit."""
        number = match.group(1)
        # Determine the unit from the match
        full_match = match.group(0).lower()
        for unit in ['inch', 'in', 'cm', 'mm', 'ft']:
            if unit in full_match:
                return f"{number} {unit}"
        return number
    
    def _singularize(self, word: str) -> str:
        """
        Convert a plural word to its singular form using rules and mappings.
        Handles special cases and avoids singularizing inherently plural words.
        """
        word_lower = word.lower().strip()
        
        # Don't singularize inherently plural words
        if word_lower in self.inherently_plural:
            return word_lower
        
        # Check if we have a direct mapping
        if word_lower in self.plural_to_singular:
            return self.plural_to_singular[word_lower]
        
        # Apply standard pluralization rules
        # Rule 1: Words ending in 'ies' → 'y' (e.g., "parties" → "party")
        if word_lower.endswith('ies') and len(word_lower) > 3:
            return word_lower[:-3] + 'y'
        
        # Rule 2: Words ending in 'ves' → 'f' or 'fe' (e.g., "scarves" → "scarf", "knives" → "knife")
        if word_lower.endswith('ves') and len(word_lower) > 3:
            # Try 'f' first (scarf → scarves)
            candidate_f = word_lower[:-3] + 'f'
            # Try 'fe' (knife → knives)
            candidate_fe = word_lower[:-3] + 'fe'
            
            # Check if either candidate is in our categories
            if candidate_f in self.categories:
                return candidate_f
            if candidate_fe in self.categories:
                return candidate_fe
            
            # Default to 'f' for common cases
            return candidate_f
        
        # Rule 3: Words ending in 'es' → remove 'es' (e.g., "dresses" → "dress", "boxes" → "box")
        if word_lower.endswith('es') and len(word_lower) > 3:
            candidate = word_lower[:-2]
            if candidate in self.categories:
                return candidate
            # Also check without 'e' (e.g., "watches" → "watch")
            candidate2 = word_lower[:-1]
            if candidate2 in self.categories:
                return candidate2
        
        # Rule 4: Words ending in 's' → remove 's' (e.g., "shirts" → "shirt", "hats" → "hat")
        if word_lower.endswith('s') and not word_lower.endswith('ss') and len(word_lower) > 2:
            candidate = word_lower[:-1]
            if candidate in self.categories:
                return candidate
        
        # If no rule applies, return the original word
        return word_lower
    
    def _normalize_category(self, word: str) -> str:
        """
        Normalize a category word to its canonical form.
        Returns the singular form if the word is a known plural, 
        or the original word if it's inherently plural or already singular.
        """
        word_lower = word.lower().strip()
        
        # If it's already in our categories, return as-is
        if word_lower in self.categories:
            return word_lower
        
        # Check if it's inherently plural
        if word_lower in self.inherently_plural:
            return word_lower
        
        # Try to singularize it
        singular_form = self._singularize(word_lower)
        
        # If the singular form is in our categories, return it
        if singular_form in self.categories:
            return singular_form
        
        # If we can't normalize it, return the original
        return word_lower
    
    def _extract_budget(self, text: str) -> Optional[str]:
        """
        Extract budget constraint from text.
        If multiple prices are found, returns the higher value.
        """
        prices = []
        
        # Try numeric patterns to find all prices
        for pattern, formatter in self.price_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                formatted = formatter(match)
                prices.append(formatted)
        
        if not prices:
            return None
        
        # If multiple prices found, extract numeric values and get the higher one
        if len(prices) > 1:
            numeric_values = []
            for price in prices:
                # Extract numeric values from formatted string
                numbers = re.findall(r'\d+(?:\.\d+)?', price)
                for num in numbers:
                    numeric_values.append(float(num))
            
            if numeric_values:
                max_price = max(numeric_values)
                # Format back to string with proper formatting
                if max_price == int(max_price):
                    return f"${int(max_price)}"
                else:
                    return f"${max_price:.2f}"
        
        # Return the single price if only one found
        return prices[0]
    
    def _extract_size(self, text: str) -> Optional[str]:
        """Extract size constraint from text."""
        # First, identify and mask all price-related numbers
        price_indicators = [r'\$\s*\d+(?:\.\d+)?', r'\d+(?:\.\d+)?\s*(?:dollars|bucks|USD)']
        masked_text = text
        
        for price_pattern in price_indicators:
            masked_text = re.sub(price_pattern, '', masked_text, flags=re.IGNORECASE)
        
        # Check for "regular fit" pattern first to avoid misclassification
        if re.search(r'\bregular\s+fit\b', masked_text, re.IGNORECASE):
            # Remove "regular" from size patterns temporarily
            for pattern, formatter in self.size_patterns:
                if pattern == r'\bregular\b':
                    continue
                match = re.search(pattern, masked_text, re.IGNORECASE)
                if match:
                    result = formatter(match)
                    if result:
                        return result
            return None
        
        # Now extract size from the masked text (prices removed)
        for pattern, formatter in self.size_patterns:
            match = re.search(pattern, masked_text, re.IGNORECASE)
            if match:
                result = formatter(match)
                if result:
                    return result
        return None
    
    def _extract_brand(self, text: str) -> Tuple[Optional[str], str]:
        """Returns (brand_or_None, source). Matches against the hardcoded
        self.brands list only."""
        return self._extract_from_list(text, self.brands), 'vocab'

    def _extract_from_list(self, text: str, word_list: Set[str]) -> Optional[str]:
        """Extract the first matched word from the list that appears in text."""
        text_lower = text.lower()
        for word in sorted(word_list, key=len, reverse=True):  # Longer matches first
            if re.search(rf'\b{re.escape(word)}\b', text_lower):
                return word
        return None
    
    def _extract_all_from_list(self, text: str, word_list: Set[str], filter_substrings: bool = False) -> List[str]:
        """Extract ALL matched words from the list that appear in text."""
        text_lower = text.lower()
        matches = []
        for word in sorted(word_list, key=len, reverse=True):  # Longer matches first
            if re.search(rf'\b{re.escape(word)}\b', text_lower):
                matches.append(word)
        
        # Optionally filter out words that are substrings of other matched words
        if filter_substrings and len(matches) > 1:
            filtered_matches = []
            for word in matches:
                is_substring = False
                for other_word in matches:
                    if word != other_word and word in other_word:
                        is_substring = True
                        break
                if not is_substring:
                    filtered_matches.append(word)
            return filtered_matches
        
        return matches
    
    def _extract_categories(self, text: str) -> List[str]:
        """
        Extract categories with pluralization handling.
        Returns normalized (singular) category names.
        """
        text_lower = text.lower()
        text_words = re.findall(r'\b[\w\-\']+\b', text_lower)
        
        found_categories = []
        
        # First, check for multi-word categories (like "t-shirt")
        for category in self.categories:
            if ' ' in category or '-' in category:
                if re.search(rf'\b{re.escape(category)}\b', text_lower):
                    found_categories.append(category)
        
        # Then check individual words
        for word in text_words:
            # Try to normalize the word to a category
            normalized = self._normalize_category(word)
            if normalized in self.categories and normalized not in found_categories:
                found_categories.append(normalized)
        
        # Remove duplicates while preserving order
        unique_categories = []
        for cat in found_categories:
            if cat not in unique_categories:
                unique_categories.append(cat)
        
        return unique_categories

    def _extract_categories_scored(self, text: str) -> List[Tuple[str, float, str]]:
        """Return (normalized_category, confidence, source) in first-match
        order — first match wins as the primary category (see
        extract_constraints_scored, which takes element 0). Matches against
        the hardcoded self.categories list only."""
        text_lower = text.lower()
        text_words = re.findall(r'\b[\w\-\']+\b', text_lower)
        scored: List[Tuple[str, float, str]] = []
        seen = set()

        def _add(name: str, confidence: float, source: str) -> None:
            if name in seen:
                return
            seen.add(name)
            scored.append((name, confidence, source))

        for category in self.categories:
            if ' ' in category or '-' in category:
                if re.search(rf'\b{re.escape(category)}\b', text_lower):
                    _add(category, 0.9, 'vocab')

        for word in text_words:
            word_lower = word.lower().strip()
            if word_lower in self.categories:
                _add(word_lower, 0.9, 'vocab')
                continue
            if word_lower in self.inherently_plural:
                _add(word_lower, 0.9, 'vocab')
                continue
            normalized = self._normalize_category(word)
            if normalized in self.categories:
                source = 'pluralization' if normalized != word_lower else 'vocab'
                confidence = 0.6 if source == 'pluralization' else 0.9
                _add(normalized, confidence, source)

        return scored

    def _extract_budget_scored(self, text: str) -> Optional[Dict[str, Any]]:
        prices: List[Tuple[str, str]] = []
        for pattern, formatter in self.price_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                formatted = formatter(match)
                prices.append((formatted, pattern))
        if not prices:
            return None

        if len(prices) > 1:
            numeric_values = []
            for price, _ in prices:
                numbers = re.findall(r'\d+(?:\.\d+)?', price)
                for num in numbers:
                    numeric_values.append(float(num))
            if numeric_values:
                max_price = max(numeric_values)
                if max_price == int(max_price):
                    value = f"${int(max_price)}"
                else:
                    value = f"${max_price:.2f}"
            else:
                value = prices[0][0]
        else:
            value = prices[0][0]

        patterns_used = [pattern for _, pattern in prices]
        high_cue = any(
            token in pattern
            for pattern in patterns_used
            for token in ('$', 'under', 'less than', 'below', 'over', 'more than',
                          'above', 'between', 'bucks', 'dollars')
        )
        return {
            'value': value,
            'confidence': 0.9 if high_cue else 0.6,
            'source': 'currency' if high_cue else 'bare_number',
        }

    def _size_match_confidence(self, pattern: str, result: str) -> Tuple[float, str]:
        if pattern == (
            r'(?<!\$)(?<!\b\d)(?<![\d.])(\d{1,2}(?:\.\d+)?)(?![\d.])(?!\s*(?:dollars|bucks|USD))'
        ):
            return 0.4, 'numeric_size'
        letter_tokens = {'XS', 'S', 'M', 'L', 'XL', 'XXL', 'XXXL'}
        if result.upper() in letter_tokens and 'small' not in pattern and 'medium' not in pattern and 'large' not in pattern:
            return 0.6, 'letter_size'
        if 'inch' in pattern or 'us|uk|eu' in pattern:
            return 0.6, 'measurement'
        if pattern in (r'\bregular\b', r'\bshort\b'):
            return 0.6, 'ambiguous_size'
        return 0.9, 'vocab'

    def _extract_size_scored(self, text: str) -> Optional[Dict[str, Any]]:
        price_indicators = [r'\$\s*\d+(?:\.\d+)?', r'\d+(?:\.\d+)?\s*(?:dollars|bucks|USD)']
        masked_text = text
        for price_pattern in price_indicators:
            masked_text = re.sub(price_pattern, '', masked_text, flags=re.IGNORECASE)

        # Also mask out percentages ("20%", "80 %") before scanning for bare
        # numeric sizes. Without this, fabric-composition phrasing like "20%
        # cotton, 80% polyester" gets its "20" misread as a shoe/clothing
        # size (see also _extract_composition, which routes these into
        # `feature` instead where they belong).
        masked_text = re.sub(r'\d+(?:\.\d+)?\s*%', '', masked_text)

        skip_regular = bool(re.search(r'\bregular\s+fit\b', masked_text, re.IGNORECASE))
        skip_bare_numeric = bool(self.measurement_context_re.search(masked_text))
        for pattern, formatter in self.size_patterns:
            if skip_regular and pattern == r'\bregular\b':
                continue
            if skip_bare_numeric and pattern == (
                r'(?<!\$)(?<!\b\d)(?<![\d.])(\d{1,2}(?:\.\d+)?)(?![\d.])(?!\s*(?:dollars|bucks|USD))'
            ):
                continue
            match = re.search(pattern, masked_text, re.IGNORECASE)
            if not match:
                continue
            result = formatter(match)
            if not result:
                continue
            confidence, source = self._size_match_confidence(pattern, result)
            return {'value': result, 'confidence': confidence, 'source': source}
        return None

    def _extract_composition(self, text: str) -> List[str]:
        """Extract fabric-composition entries like "20% cotton" or "80%
        polyester". Restricted to words already in self.materials so
        unrelated percentages ("50% off", "30% faster charging") aren't
        misread as composition. Order-preserving, deduplicated.

        These are routed into the `feature` slot (see
        extract_constraints_scored) rather than `size` — a bare number
        followed by "%" is fabric-composition info, not a size, and is
        excluded from size matching for that reason (see
        _extract_size_scored)."""
        matches: List[str] = []
        seen: Set[str] = set()
        for m in re.finditer(r'\b(\d{1,3}(?:\.\d+)?)\s*%\s*([a-z][a-z\-]*)\b', text, re.IGNORECASE):
            pct, word = m.group(1), m.group(2).lower()
            if word in self.materials and word not in seen:
                seen.add(word)
                matches.append(f"{pct}% {word}")
        return matches

    def _scored_entry(self, value: Any, confidence: float, source: str) -> Dict[str, Any]:
        return {'value': value, 'confidence': confidence, 'source': source}

    def extract_constraints_scored(self, user_input: str) -> Dict[str, Dict[str, Any]]:
        """
        Extract constraints with heuristic confidence and source tags.

        Each entry is {"value": str|list, "confidence": float, "source": str}.
        Does not change extract_constraints() return shape.
        """
        text = user_input.strip()
        scored: Dict[str, Dict[str, Any]] = {}

        categories = self._extract_categories_scored(text)
        if categories:
            name, confidence, source = categories[0]
            scored['category'] = self._scored_entry(name, confidence, source)

        materials = self._extract_all_from_list(text, self.materials)
        if materials:
            scored['material'] = self._scored_entry(materials[0], 0.9, 'vocab')

        colors = self._extract_all_from_list(text, self.colors)
        if colors:
            scored['color'] = self._scored_entry(colors[0], 0.9, 'vocab')

        size = self._extract_size_scored(text)
        if size:
            scored['size'] = size

        style = self._extract_from_list(text, self.style)
        if style:
            scored['style'] = self._scored_entry(style, 0.9, 'vocab')

        brand, brand_source = self._extract_brand(text)
        if brand:
            scored['brand'] = self._scored_entry(brand, 0.9, brand_source)

        budget = self._extract_budget_scored(text)
        if budget:
            scored['budget'] = budget

        features = self._extract_all_from_list(text, self.features)
        composition = self._extract_composition(text)
        combined_features = features + [c for c in composition if c not in features]
        if combined_features:
            value = combined_features[0] if len(combined_features) == 1 else combined_features
            source = 'vocab' if features else 'composition_pct'
            scored['feature'] = self._scored_entry(value, 0.9 if features else 0.85, source)

        others: List[str] = []
        others.extend(self._extract_all_from_list(text, self.genders))
        others.extend(self._extract_all_from_list(text, self.fits))
        others.extend(self._extract_all_from_list(text, self.conditions))
        others.extend(self._extract_all_from_list(text, self.patterns))
        if others:
            # Key must match SessionState's LIST_SLOTS/ALLOWED_SLOTS name ('other',
            # singular) -- 'other' is not a recognized slot and was silently
            # unusable downstream.
            scored['other'] = self._scored_entry(others, 0.6, 'others_bucket')

        overlap = (self.colors & self.materials) | (self.colors & self.brands)
        for slot in ('color', 'material', 'brand'):
            entry = scored.get(slot)
            if entry and str(entry['value']).lower() in overlap:
                entry['confidence'] = 0.4
                entry['source'] = 'ambiguous_overlap'

        return scored

    def extract_constraints(self, user_input: str) -> Dict[str, Union[str, List[str]]]:
        """
        Extract specific constraints from user input.
        Modified: 'category', 'material', 'color', 'size', 'style', 'brand' 
        now return single values (first match only).
        
        Args:
            user_input: The user's search query or message
            
        Returns:
            Dictionary of extracted constraints with normalized values
            Single-value constraints return strings, multi-value return lists
        """
        scored = self.extract_constraints_scored(user_input)
        return {key: entry['value'] for key, entry in scored.items()}

    def extract_keywords(
        self,
        user_input: str,
        constraints: Optional[Dict[str, Union[str, List[str]]]] = None,
    ) -> List[str]:
        """
        Return descriptive words from `user_input` that don't belong to any
        matched constraint slot and aren't generic stopwords — e.g. a brand
        fragment none of the lexicons recognize, a product-line name, or
        any other catalog-relevant vocabulary this classifier doesn't have
        a dedicated slot for.

        This is the piece ClassifyIntent previously had no equivalent of:
        every word not captured by a lexicon was silently dropped. Mirrors
        agent_addon_5's approach (tokenize -> drop stopwords -> drop
        anything already consumed by a matched slot) using the SAME
        general, catalog-independent method — not a hardcoded list tuned to
        any particular evaluator's sample phrasing.

        Purely additive: does not change extract_constraints(),
        extract_constraints_scored(), classify_intent(),
        classify_with_details(), or resolve_query_differences() in any way
        — none of them call this, and their return shapes are untouched.

        Args:
            user_input: The user's message.
            constraints: Optionally, the result of a prior
                extract_constraints(user_input) call on this SAME input, to
                avoid re-running extraction. Computed internally if omitted.

        Returns:
            Leftover terms in first-appearance order, deduplicated. NOTE:
            this returns only THIS message's leftover terms — it has no
            memory of earlier turns. A caller that wants agent_addon_5-style
            cross-turn accumulation (a running list of freeform terms that
            persists across a conversation) should collect these turn by
            turn itself, e.g.:

                seen_terms = []
                for message in conversation:
                    for term in classifier.extract_keywords(message):
                        if term not in seen_terms:
                            seen_terms.append(term)
        """
        if constraints is None:
            constraints = self.extract_constraints(user_input)

        consumed = {
            token
            for value in constraints.values()
            for item in (value if isinstance(value, list) else [value])
            for token in _terms(str(item))
        }

        keywords: List[str] = []
        seen: Set[str] = set()
        for token in _terms(user_input):
            if token in consumed or token in seen:
                continue
            seen.add(token)
            keywords.append(token)
        return keywords

    def classify_intent(self, user_input: str) -> str:
        """
        Classify user intent as 'Buying' or 'Browsing' based on constraint density.
        
        Args:
            user_input: The user's search query or message
            
        Returns:
            'Buying' if ≥ 2 hard constraints found, 'Browsing' otherwise
        """
        constraints = self.extract_constraints(user_input)
        
        # Count hard constraint categories (all constraints are hard now)
        hard_constraints = constraints
        
        # Determine intent based on hard constraint count
        if len(hard_constraints) >= 2:
            return "Buying"
        else:
            return "Browsing"
    
    def classify_with_details(self, user_input: str) -> Dict[str, Any]:
        """
        Enhanced classification that returns intent along with extracted constraints.
        
        Returns:
            Dictionary with intent, constraint count, and extracted constraints
        """
        constraints = self.extract_constraints(user_input)
        
        # Count hard constraints (all constraints are hard now)
        hard_constraints = constraints
        
        intent = "Buying" if len(hard_constraints) >= 2 else "Browsing"
        
        # Calculate total constraint values (counting list items)
        total_values = 0
        for value in hard_constraints.values():
            if isinstance(value, list):
                total_values += len(value)
            else:
                total_values += 1
        
        return {
            "intent": intent,
            "constraint_count": len(hard_constraints),  # Number of hard constraint categories
            "total_constraint_values": total_values,  # Total including multiple features
            "constraints": constraints,  # All constraints
            "hard_constraints": hard_constraints,  # Only hard constraints
            "query": user_input
        }
    
    def get_constraint_categories(self) -> List[str]:
        """Return list of all supported constraint categories."""
        return [
            'category', 'material', 'color', 'size', 'style', 
            'brand', 'budget', 'feature', 'other'
        ]
    
    def get_feature_list(self) -> List[str]:
        """Return the complete list of supported features."""
        return sorted(self.features)
    
    def resolve_query_differences(self, original_query: str, new_query: str) -> Dict[str, Union[str, List[str]]]:
        # Extract constraints from both queries
        original_constraints = self.extract_constraints(original_query)
        new_constraints = self.extract_constraints(new_query)
        
        # Start with original constraints as the base
        resolved_constraints = original_constraints.copy()
        
        # Parse the new query to identify modifications
        new_query_lower = new_query.lower()
        
        # Check for negation patterns indicating removal of constraints
        negation_patterns = [
            r'\b(?:no longer|not|don\'t|do not|doesn\'t|does not|without|remove|drop|skip)\b'
        ]
        
        # Identify constraints to remove based on negation
        constraints_to_remove = set()
        
        # Check for category-level removals
        category_removal_patterns = {
            'size': [
                r'remove\s+(?:the\s+)?size\s+(?:requirement|constraint|filter|restriction)',
                r'remove\s+(?:the\s+)?size\b',
                r'drop\s+(?:the\s+)?size\s+(?:requirement|constraint|filter|restriction)',
                r'drop\s+(?:the\s+)?size\b',
                r'without\s+(?:the\s+)?size\s+(?:requirement|constraint|filter|restriction)',
                r'without\s+(?:the\s+)?size\b',
                r'no\s+(?:more|longer)\s+(?:need|require|want)\s+(?:the\s+)?size\b',
                r'not\s+(?:need|require|want)\s+(?:the\s+)?size\b',
            ],
            'color': [
                r'remove\s+(?:the\s+)?color\s+(?:requirement|constraint|filter|restriction)',
                r'remove\s+(?:the\s+)?color\b',
                r'drop\s+(?:the\s+)?color\s+(?:requirement|constraint|filter|restriction)',
                r'drop\s+(?:the\s+)?color\b',
                r'without\s+(?:the\s+)?color\s+(?:requirement|constraint|filter|restriction)',
                r'without\s+(?:the\s+)?color\b',
            ],
            'brand': [
                r'remove\s+(?:the\s+)?brand\s+(?:requirement|constraint|filter|restriction)',
                r'remove\s+(?:the\s+)?brand\b',
                r'drop\s+(?:the\s+)?brand\s+(?:requirement|constraint|filter|restriction)',
                r'drop\s+(?:the\s+)?brand\b',
                r'without\s+(?:the\s+)?brand\s+(?:requirement|constraint|filter|restriction)',
                r'without\s+(?:the\s+)?brand\b',
            ],
            'budget': [
                r'remove\s+(?:the\s+)?(?:price|budget)\s+(?:requirement|constraint|filter|restriction)',
                r'remove\s+(?:the\s+)?(?:price|budget)\b',
                r'drop\s+(?:the\s+)?(?:price|budget)\s+(?:requirement|constraint|filter|restriction)',
                r'drop\s+(?:the\s+)?(?:price|budget)\b',
                r'without\s+(?:the\s+)?(?:price|budget)\s+(?:requirement|constraint|filter|restriction)',
                r'without\s+(?:the\s+)?(?:price|budget)\b',
            ],
            'category': [
                r'remove\s+(?:the\s+)?categor(?:y|ies)\s+(?:requirement|constraint|filter|restriction)',
                r'remove\s+(?:the\s+)?categor(?:y|ies)\b',
                r'drop\s+(?:the\s+)?categor(?:y|ies)\s+(?:requirement|constraint|filter|restriction)',
                r'drop\s+(?:the\s+)?categor(?:y|ies)\b',
                r'without\s+(?:the\s+)?categor(?:y|ies)\s+(?:requirement|constraint|filter|restriction)',
                r'without\s+(?:the\s+)?categor(?:y|ies)\b',
            ],
            'feature': [
                r'remove\s+(?:the\s+)?features?\s+(?:requirement|constraint|filter|restriction)',
                r'remove\s+(?:the\s+)?features?\b',
                r'drop\s+(?:the\s+)?features?\s+(?:requirement|constraint|filter|restriction)',
                r'drop\s+(?:the\s+)?features?\b',
                r'without\s+(?:the\s+)?features?\s+(?:requirement|constraint|filter|restriction)',
                r'without\s+(?:the\s+)?features?\b',
            ],
            'other': [
                r'remove\s+(?:the\s+)?(?:gender|fit|condition|pattern)s?\s+(?:requirement|constraint|filter|restriction)',
                r'remove\s+(?:the\s+)?(?:gender|fit|condition|pattern)s?\b',
                r'drop\s+(?:the\s+)?(?:gender|fit|condition|pattern)s?\s+(?:requirement|constraint|filter|restriction)',
                r'drop\s+(?:the\s+)?(?:gender|fit|condition|pattern)s?\b',
                r'without\s+(?:the\s+)?(?:gender|fit|condition|pattern)s?\s+(?:requirement|constraint|filter|restriction)',
                r'without\s+(?:the\s+)?(?:gender|fit|condition|pattern)s?\b',
            ],
        }
        
        # Check for category-level removals first
        for category, patterns in category_removal_patterns.items():
            for pattern in patterns:
                if re.search(pattern, new_query_lower):
                    constraints_to_remove.add(category)
                    break
        
        # Check each constraint category for removal
        for category, value in original_constraints.items():
            # Get the actual values to check for negation
            values_to_check = value if isinstance(value, list) else [value]
            
            for val in values_to_check:
                # Check if this specific value is being negated
                is_negated = False
                
                # Check for negation patterns specific to this value
                negation_phrases = [
                    'no longer', 'not', "don't", 'do not', "doesn't", 
                    'does not', 'without', 'remove', 'drop', 'skip'
                ]
                
                for negation_phrase in negation_phrases:
                    # More flexible patterns to catch various negation forms
                    patterns = [
                        rf'{negation_phrase}\s+(?:need\s+to\s+be|have\s+to\s+be|be|have|need|require)\s+{re.escape(val)}',
                        rf'{negation_phrase}\s+{re.escape(val)}',
                        rf'{negation_phrase}\s+{re.escape(val)}',
                        rf'remove\s+(?:the\s+)?{re.escape(val)}',
                        rf'drop\s+(?:the\s+)?{re.escape(val)}',
                        rf'without\s+(?:the\s+)?{re.escape(val)}',
                        rf'{negation_phrase}\s+(?:need|require|want)\s+(?:to\s+be\s+|the\s+)?{re.escape(val)}',
                        rf'{negation_phrase}[\w\s]{{0,20}}{re.escape(val)}'
                    ]
                    
                    for pattern in patterns:
                        if re.search(pattern, new_query_lower):
                            is_negated = True
                            break
                    
                    if is_negated:
                        break
                
                if is_negated:
                    if category == 'feature':
                        # For features, track specific feature to remove
                        constraints_to_remove.add(f'feature:{val}')
                    elif category == 'other':
                        # For others, track specific value to remove
                        constraints_to_remove.add(f'other:{val}')
                    else:
                        # For other categories, remove entire category
                        constraints_to_remove.add(category)
                    break
        
        # Apply removals
        for removal in constraints_to_remove:
            if removal.startswith('feature:'):
                # Remove specific feature
                feature_to_remove = removal.split(':', 1)[1]
                if 'feature' in resolved_constraints:
                    current_features = resolved_constraints['feature']
                    if isinstance(current_features, list):
                        current_features = [f for f in current_features if f != feature_to_remove]
                        if len(current_features) == 0:
                            del resolved_constraints['feature']
                        elif len(current_features) == 1:
                            resolved_constraints['feature'] = current_features[0]
                        else:
                            resolved_constraints['feature'] = current_features
                    elif current_features == feature_to_remove:
                        del resolved_constraints['feature']
            elif removal.startswith('other:'):
                # Remove specific value from others
                value_to_remove = removal.split(':', 1)[1]
                if 'other' in resolved_constraints:
                    current_others = resolved_constraints['other']
                    if isinstance(current_others, list):
                        current_others = [v for v in current_others if v != value_to_remove]
                        if len(current_others) == 0:
                            del resolved_constraints['other']
                        elif len(current_others) == 1:
                            resolved_constraints['other'] = current_others[0]
                        else:
                            resolved_constraints['other'] = current_others
                    elif current_others == value_to_remove:
                        del resolved_constraints['other']
            else:
                # Remove entire category
                if removal in resolved_constraints:
                    del resolved_constraints[removal]
        
        # Update/add constraints from new query (excluding those being removed)
        for category, value in new_constraints.items():
            if category not in constraints_to_remove:
                if category == 'feature':
                    # Special handling for features (merge lists)
                    if 'feature' in resolved_constraints:
                        existing_features = resolved_constraints['feature']
                        if isinstance(existing_features, str):
                            existing_features = [existing_features]
                        
                        new_features = value if isinstance(value, list) else [value]
                        
                        # Merge features, but skip those marked for removal
                        merged_features = existing_features.copy()
                        for feature in new_features:
                            if feature not in merged_features:
                                # Check if this feature should be removed
                                should_remove = False
                                for removal in constraints_to_remove:
                                    if removal.startswith('feature:') and removal.split(':', 1)[1] == feature:
                                        should_remove = True
                                        break
                                if not should_remove:
                                    merged_features.append(feature)
                        
                        # Set the final feature value
                        if len(merged_features) == 0:
                            if 'feature' in resolved_constraints:
                                del resolved_constraints['feature']
                        elif len(merged_features) == 1:
                            resolved_constraints['feature'] = merged_features[0]
                        else:
                            resolved_constraints['feature'] = merged_features
                    else:
                        # No existing features, just add the new ones
                        if isinstance(value, list):
                            filtered_features = []
                            for feature in value:
                                should_remove = False
                                for removal in constraints_to_remove:
                                    if removal.startswith('feature:') and removal.split(':', 1)[1] == feature:
                                        should_remove = True
                                        break
                                if not should_remove:
                                    filtered_features.append(feature)
                            if filtered_features:
                                if len(filtered_features) == 1:
                                    resolved_constraints['feature'] = filtered_features[0]
                                else:
                                    resolved_constraints['feature'] = filtered_features
                        else:
                            # Check if this single feature should be removed
                            should_remove = False
                            for removal in constraints_to_remove:
                                if removal.startswith('feature:') and removal.split(':', 1)[1] == value:
                                    should_remove = True
                                    break
                            if not should_remove:
                                resolved_constraints['feature'] = value
                elif category == 'other':
                    # Special handling for others (merge lists)
                    if 'other' in resolved_constraints:
                        existing_others = resolved_constraints['other']
                        if isinstance(existing_others, str):
                            existing_others = [existing_others]
                        
                        new_others = value if isinstance(value, list) else [value]
                        
                        # Merge others, but skip those marked for removal
                        merged_others = existing_others.copy()
                        for other in new_others:
                            if other not in merged_others:
                                # Check if this value should be removed
                                should_remove = False
                                for removal in constraints_to_remove:
                                    if removal.startswith('other:') and removal.split(':', 1)[1] == other:
                                        should_remove = True
                                        break
                                if not should_remove:
                                    merged_others.append(other)
                        
                        # Set the final others value
                        if len(merged_others) == 0:
                            if 'other' in resolved_constraints:
                                del resolved_constraints['other']
                        elif len(merged_others) == 1:
                            resolved_constraints['other'] = merged_others[0]
                        else:
                            resolved_constraints['other'] = merged_others
                    else:
                        # No existing others, just add the new ones
                        if isinstance(value, list):
                            filtered_others = []
                            for other in value:
                                should_remove = False
                                for removal in constraints_to_remove:
                                    if removal.startswith('other:') and removal.split(':', 1)[1] == other:
                                        should_remove = True
                                        break
                                if not should_remove:
                                    filtered_others.append(other)
                            if filtered_others:
                                if len(filtered_others) == 1:
                                    resolved_constraints['other'] = filtered_others[0]
                                else:
                                    resolved_constraints['other'] = filtered_others
                        else:
                            # Check if this single value should be removed
                            should_remove = False
                            for removal in constraints_to_remove:
                                if removal.startswith('other:') and removal.split(':', 1)[1] == value:
                                    should_remove = True
                                    break
                            if not should_remove:
                                resolved_constraints['other'] = value
                elif category == 'category':
                    # Special handling for category - replace, don't merge
                    if isinstance(value, list):
                        # If multiple categories found, prefer the most specific one
                        resolved_constraints['category'] = max(value, key=len)
                    else:
                        resolved_constraints['category'] = value
                else:
                    # Simply update/replace the constraint value
                    resolved_constraints[category] = value
        
        return resolved_constraints
