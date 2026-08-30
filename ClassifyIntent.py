import re
from typing import List, Set, Dict, Optional, Any, Tuple, Union

class IntentClassifier:
    """
    Rule-based intent router that distinguishes between 'Buying' (high constraint density)
    and 'Browsing' (low constraint density) user queries.
    Supports multiple features and multi-value constraints.
    """
    
    def __init__(self):
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
            'alloy', 'chromium', 'titanium', 'tungsten', 'copper', 'brass',
            'bronze', 'pewter', 'nickel', 'cobalt', 'zinc', 'iron', 'aluminum'
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
            'necklace', 'chain', 'jewelry', 'bracelet', 'pendant',
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
            'bracelet': 'bracelets',
            'pendant': 'pendants',
        }
        
        # Define plural-to-singular mapping (reverse of above, for normalization)
        self.plural_to_singular = {v: k for k, v in self.singular_to_plural.items()}
        
        # Words that are inherently plural (should not be singularized)
        self.inherently_plural = {
            'pants', 'jeans', 'shorts', 'leggings', 'sunglasses', 
            'earrings', 'studs', 'socks', 'shoes', 'boots', 'sandals',
            'sneakers', 'heels', 'gloves', 'jewelry'
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
            'multi-functional', 'all-in-one', 'compact', 'space saving'
        }
        
        # Size mappings for normalization
        self.size_mappings = {
            'extra small': 'XS', 'small': 'S', 'medium': 'M', 'large': 'L',
            'x-large': 'XL', 'xx-large': 'XXL', 'xxx-large': 'XXXL',
            'extra large': 'XL', 'extra small': 'XS', '3XL': 'XXXL', '2XL': 'XXL'
        }
        
        # Phrases that indicate key considerations or main points
        self.key_phrase_patterns = [
            r'key considerations?\s*(?:are|is)?\s*:?',
            r'main points?\s*(?:are|is)?\s*:?',
            r'what matters\s*(?:is|are)?\s*:?',
            r'important (?:things?|factors?|aspects?)\s*(?:are|is)?\s*:?',
            r'key (?:things?|factors?|aspects?|requirements?|features?)\s*(?:are|is)?\s*:?',
            r'essential (?:things?|factors?|aspects?)\s*(?:are|is)?\s*:?',
            r'critical (?:things?|factors?|aspects?)\s*(?:are|is)?\s*:?',
            r'must[- ]haves?\s*(?:are|is)?\s*:?',
            r'priorities?\s*(?:are|is)?\s*:?',
            r'focus (?:on|areas?)\s*(?:are|is)?\s*:?',
            r'requirements?\s*(?:are|is)?\s*:?',
            r'it must be\s*:?',
            r'must be\s*:?',
            r'should be\s*:?',
            r'needs to be\s*:?',
            r'has to be\s*:?',
        ]
        
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
        
        # Compile key phrase patterns
        self.compiled_key_phrase_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.key_phrase_patterns
        ]
        
        # Patterns for extracting constraint values after key phrases
        self.constraint_extraction_patterns = [
            # Pattern for "Material: X" or "material: X"
            (r'\b(?:material|fabric|made of)\s*:\s*([^,.]+)', 'material'),
            # Pattern for "Color: X" or "color: X"
            (r'\b(?:color|colour)\s*:\s*([^,.]+)', 'color'),
            # Pattern for "Size: X" or "size: X"
            (r'\b(?:size|sizing)\s*:\s*([^,.]+)', 'size'),
            # Pattern for "Brand: X" or "brand: X"
            (r'\b(?:brand|make)\s*:\s*([^,.]+)', 'brand'),
            # Pattern for "Style: X" or "style: X"
            (r'\b(?:style|type|kind)\s*:\s*([^,.]+)', 'style'),
            # Pattern for "Price: X" or "price: X" or "budget: X"
            (r'\b(?:price|budget|cost)\s*:\s*([^,.]+)', 'budget'),
            # Pattern for "Feature: X" or "feature: X"
            (r'\b(?:feature|features?|with)\s*:\s*([^,.]+)', 'feature'),
            # Pattern for "Category: X" or "category: X"
            (r'\b(?:category|type of)\s*:\s*([^,.]+)', 'category'),
            # Pattern for other attribute: value pairs
            (r'\b([a-z]+(?:\s+[a-z]+)*)\s*:\s*([^,.]+)', 'other'),
        ]
        
        # Compile constraint extraction patterns
        self.compiled_constraint_patterns = [
            (re.compile(pattern, re.IGNORECASE), category) 
            for pattern, category in self.constraint_extraction_patterns
        ]
    
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
    
    def _extract_key_phrase_constraints(self, text: str) -> List[str]:
        """
        Extract constraint values from key phrases in the text.
        This method identifies key phrase indicators and extracts the actual
        constraint values that follow them (e.g., "Material: alloy" → "Material: alloy").
        """
        constraints_found = []
        text_lower = text.lower()
        
        # Find all positions where key phrases occur
        key_phrase_positions = []
        for pattern in self.compiled_key_phrase_patterns:
            for match in pattern.finditer(text_lower):
                key_phrase_positions.append((match.start(), match.end()))
        
        # Sort positions by start index
        key_phrase_positions.sort(key=lambda x: x[0])
        
        # Extract content after each key phrase
        for start, end in key_phrase_positions:
            # Get the content after the key phrase
            content_after = text[end:].strip()
            
            if not content_after:
                continue
            
            # Try to extract structured constraint values (e.g., "Material: alloy")
            extracted_structured = self._extract_structured_constraints(content_after)
            if extracted_structured:
                constraints_found.extend(extracted_structured)
            else:
                # If no structured constraints found, try to extract simple values
                # Split by common separators
                items = re.split(r'[,;]|\band\b|\bor\b', content_after)
                for item in items:
                    item = item.strip()
                    if item and len(item) > 1:
                        # Check if the item itself contains a constraint pattern
                        item_constraints = self._extract_structured_constraints(item)
                        if item_constraints:
                            constraints_found.extend(item_constraints)
                        elif len(item.split()) <= 3:  # Only add short phrases as others
                            constraints_found.append(item)
        
        # Remove duplicates while preserving order
        unique_constraints = []
        for constraint in constraints_found:
            if constraint not in unique_constraints:
                unique_constraints.append(constraint)
        
        return unique_constraints
    
    def _extract_structured_constraints(self, text: str) -> List[str]:
        """
        Extract structured constraint values from text (e.g., "Material: alloy").
        Returns a list of constraint strings in the format "category: value".
        """
        structured_constraints = []
        text_lower = text.lower()
        
        # Try each constraint extraction pattern
        for pattern, category in self.compiled_constraint_patterns:
            for match in pattern.finditer(text_lower):
                if category == 'other':
                    # For generic patterns, check if the key is a known constraint type
                    key = match.group(1).strip()
                    value = match.group(2).strip()
                    
                    # Map common keys to constraint categories
                    key_mapping = {
                        'material': 'material',
                        'fabric': 'material',
                        'made of': 'material',
                        'color': 'color',
                        'colour': 'color',
                        'size': 'size',
                        'sizing': 'size',
                        'brand': 'brand',
                        'make': 'brand',
                        'style': 'style',
                        'type': 'style',
                        'kind': 'style',
                        'price': 'budget',
                        'budget': 'budget',
                        'cost': 'budget',
                        'feature': 'feature',
                        'features': 'feature',
                        'with': 'feature',
                        'category': 'category',
                        'type of': 'category',
                    }
                    
                    if key in key_mapping:
                        mapped_category = key_mapping[key]
                        structured_constraints.append(f"{mapped_category}: {value}")
                    else:
                        # For unknown keys, just add as generic
                        structured_constraints.append(f"{key}: {value}")
                else:
                    # For specific patterns, extract the value
                    value = match.group(1).strip()
                    structured_constraints.append(f"{category}: {value}")
        
        # Also check for patterns without explicit key phrases
        # For example, "must be material: chromium" or "it must be chromium"
        must_be_patterns = [
            (r'(?:must|should|needs to|has to)\s+be\s+(?:made of\s+)?([^,.]+)', 'material'),
            (r'(?:must|should|needs to|has to)\s+be\s+([^,.]+)', 'other'),
        ]
        
        for pattern, category in must_be_patterns:
            for match in re.finditer(pattern, text_lower):
                value = match.group(1).strip()
                if category == 'material':
                    # Check if the value is a known material
                    if value in self.materials:
                        structured_constraints.append(f"material: {value}")
                else:
                    # Check if the value matches any known constraint type
                    value_lower = value.lower()
                    if value_lower in self.materials:
                        structured_constraints.append(f"material: {value}")
                    elif value_lower in self.colors:
                        structured_constraints.append(f"color: {value}")
                    elif value_lower in self.brands:
                        structured_constraints.append(f"brand: {value}")
                    elif value_lower in self.style:
                        structured_constraints.append(f"style: {value}")
        
        return structured_constraints
    
    def extract_constraints(self, user_input: str) -> Dict[str, Union[str, List[str]]]:
        """
        Extract specific constraints from user input.
        Modified: 'category', 'material', 'color', 'size', 'style', 'brand' 
        now return single values (first match only).
        Key phrases like "key considerations are:" are placed under 'others'.
        
        Args:
            user_input: The user's search query or message
            
        Returns:
            Dictionary of extracted constraints with normalized values
            Single-value constraints return strings, multi-value return lists
        """
        text = user_input.strip()
        constraints = {}
        
        # Extract category with pluralization handling - NOW SINGLE VALUE
        categories = self._extract_categories(text)
        if categories:
            constraints['category'] = categories[0]  # Take first category only
        
        # Extract material - NOW SINGLE VALUE
        materials = self._extract_all_from_list(text, self.materials)
        if materials:
            constraints['material'] = materials[0]  # Take first material only
        
        # Extract color - NOW SINGLE VALUE
        colors = self._extract_all_from_list(text, self.colors)
        if colors:
            constraints['color'] = colors[0]  # Take first color only
        
        # Extract size (single value - unchanged)
        size = self._extract_size(text)
        if size:
            constraints['size'] = size
        
        # Extract style (single value - unchanged)
        style = self._extract_from_list(text, self.style)
        if style:
            constraints['style'] = style
        
        # Extract brand - NOW EXPLICITLY SINGLE VALUE
        brand = self._extract_from_list(text, self.brands)
        if brand:
            constraints['brand'] = brand
        
        # Extract budget (hard constraint - specific amount)
        budget = self._extract_budget(text)
        if budget:
            constraints['budget'] = budget
        
        # Extract features (MULTIPLE features still supported)
        features = self._extract_all_from_list(text, self.features)
        if features:
            constraints['feature'] = features[0] if len(features) == 1 else features
        
        # Extract others (genders, fits, conditions, patterns combined)
        others = []
        
        # Extract genders (can be multiple)
        genders = self._extract_all_from_list(text, self.genders)
        if genders:
            others.extend(genders)
        
        # Extract fits (can be multiple)
        fits = self._extract_all_from_list(text, self.fits)
        if fits:
            others.extend(fits)
        
        # Extract conditions (can be multiple)
        conditions = self._extract_all_from_list(text, self.conditions)
        if conditions:
            others.extend(conditions)
        
        # Extract patterns (can be multiple)
        patterns = self._extract_all_from_list(text, self.patterns)
        if patterns:
            others.extend(patterns)
        
        # Extract key phrase constraints (NEW)
        key_phrase_constraints = self._extract_key_phrase_constraints(text)
        if key_phrase_constraints:
            # Add key phrase constraints to others list
            others.extend(key_phrase_constraints)
        
        # Add others to constraints if any found
        if others:
            # Remove duplicates while preserving order
            unique_others = []
            for other in others:
                if other not in unique_others:
                    unique_others.append(other)
            constraints['others'] = unique_others
        
        return constraints
    
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
            'brand', 'budget', 'feature', 'others'
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
            'others': [
                r'remove\s+(?:the\s+)?(?:gender|fit|condition|pattern|key)s?\s+(?:requirement|constraint|filter|restriction)',
                r'remove\s+(?:the\s+)?(?:gender|fit|condition|pattern|key)s?\b',
                r'drop\s+(?:the\s+)?(?:gender|fit|condition|pattern|key)s?\s+(?:requirement|constraint|filter|restriction)',
                r'drop\s+(?:the\s+)?(?:gender|fit|condition|pattern|key)s?\b',
                r'without\s+(?:the\s+)?(?:gender|fit|condition|pattern|key)s?\s+(?:requirement|constraint|filter|restriction)',
                r'without\s+(?:the\s+)?(?:gender|fit|condition|pattern|key)s?\b',
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
                # Skip key phrase prefixed values for removal checking
                if val.startswith('key:'):
                    continue
                    
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
                    elif category == 'others':
                        # For others, track specific value to remove
                        constraints_to_remove.add(f'others:{val}')
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
            elif removal.startswith('others:'):
                # Remove specific value from others
                value_to_remove = removal.split(':', 1)[1]
                if 'others' in resolved_constraints:
                    current_others = resolved_constraints['others']
                    if isinstance(current_others, list):
                        current_others = [v for v in current_others if v != value_to_remove]
                        if len(current_others) == 0:
                            del resolved_constraints['others']
                        elif len(current_others) == 1:
                            resolved_constraints['others'] = current_others[0]
                        else:
                            resolved_constraints['others'] = current_others
                    elif current_others == value_to_remove:
                        del resolved_constraints['others']
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
                elif category == 'others':
                    # Special handling for others (merge lists)
                    if 'others' in resolved_constraints:
                        existing_others = resolved_constraints['others']
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
                                    if removal.startswith('others:') and removal.split(':', 1)[1] == other:
                                        should_remove = True
                                        break
                                if not should_remove:
                                    merged_others.append(other)
                        
                        # Set the final others value
                        if len(merged_others) == 0:
                            if 'others' in resolved_constraints:
                                del resolved_constraints['others']
                        elif len(merged_others) == 1:
                            resolved_constraints['others'] = merged_others[0]
                        else:
                            resolved_constraints['others'] = merged_others
                    else:
                        # No existing others, just add the new ones
                        if isinstance(value, list):
                            filtered_others = []
                            for other in value:
                                should_remove = False
                                for removal in constraints_to_remove:
                                    if removal.startswith('others:') and removal.split(':', 1)[1] == other:
                                        should_remove = True
                                        break
                                if not should_remove:
                                    filtered_others.append(other)
                            if filtered_others:
                                if len(filtered_others) == 1:
                                    resolved_constraints['others'] = filtered_others[0]
                                else:
                                    resolved_constraints['others'] = filtered_others
                        else:
                            # Check if this single value should be removed
                            should_remove = False
                            for removal in constraints_to_remove:
                                if removal.startswith('others:') and removal.split(':', 1)[1] == value:
                                    should_remove = True
                                    break
                            if not should_remove:
                                resolved_constraints['others'] = value
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
    
#hello
