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
            'canvas', 'mesh', 'satin', 'lace', 'chiffon', 'tweed', 'corduroy', 'stainless steel', 'sterling silver',
            'platinum', 'gold', 'silver'
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
        
        # Occasion/style terms
        self.occasions: Set[str] = {
            'casual', 'formal', 'business', 'party', 'wedding', 'workout',
            'athletic', 'outdoor', 'everyday', 'dressy', 'professional',
            'vacation', 'beach', 'winter', 'summer', 'spring', 'fall'
        }
        
        # Gender/target audience
        self.genders: Set[str] = {
            'men', 'women', 'mens', 'womens', 'boys', 'girls', 'kids',
            'unisex', 'toddler', 'infant', 'baby', 'youth', 'adult', 'children'
        }
        
        # Product categories
        self.categories: Set[str] = {
            'shirt', 'pants', 'shoes', 'jacket', 'dress', 'skirt', 'shorts',
            'sweater', 'hoodie', 't-shirt', 'jeans', 'socks', 'hat', 'cap',
            'scarf', 'gloves', 'belt', 'watch', 'bag', 'backpack', 'sunglasses',
            'boots', 'sandals', 'sneakers', 'heels', 'suit', 'blazer', 'coat',
            'swimsuit', 'underwear', 'bra', 'leggings', 'cardigan', 'vest', 'earrings', 'rings', 'stud', 'necklace', 'chain'
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
        
        # Qualitative price indicators (not hard constraints)
        self.price_qualifiers = {
            'cheap': 'budget',
            'affordable': 'budget',
            'budget': 'budget',
            'inexpensive': 'budget',
            'expensive': 'premium',
            'premium': 'premium',
            'luxury': 'luxury',
            'high-end': 'luxury',
            'high end': 'luxury',
            'reasonable': 'mid-range',
            'moderate': 'mid-range',
            'mid-range': 'mid-range',
            'mid range': 'mid-range'
        }
        
        # Compile regex patterns
        self._compile_patterns()
    
    def _compile_patterns(self):#"""Pre-compile regex patterns for efficiency."""
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
            (r'\bshort\b', lambda m: "short")
        ]
        
        # Rating patterns with capture groups
        self.rating_patterns = [
            (r'\b(?:rated|rating|stars|star)\s+(\d+(?:\.\d+)?)\b', 
             lambda m: f"{m.group(1)} stars"),
            (r'\b(\d+(?:\.\d+)?)\s+stars?\b', 
             lambda m: f"{m.group(1)} stars"),
            (r'\b(\d+(?:\.\d+)?)\s*/\s*5\b', 
             lambda m: f"{m.group(1)}/5"),
            (r'\bhighly\s+rated\b', lambda m: ">4.0 stars"),
            (r'\btop\s+rated\b', lambda m: ">4.5 stars"),
            (r'\bbest\s+seller\b', lambda m: "bestseller"),
            (r'\bbestseller\b', lambda m: "bestseller")
        ]
    
    def _normalize_size(self, size: str) -> str:#"""Normalize size representations.(e.g., "extra large" → "XL")"""
        """Normalize size representations.(e.g., "extra large" → "XL")"""
        size_lower = size.lower().strip()
        if size_lower in self.size_mappings:
            return self.size_mappings[size_lower]
        return size.upper()
    
    def _extract_measurement(self, match) -> str:#"""Extract measurement with unit."""
        """Extract measurement with unit."""
        number = match.group(1)
        # Determine the unit from the match
        full_match = match.group(0).lower()
        for unit in ['inch', 'in', 'cm', 'mm', 'ft']:
            if unit in full_match:
                return f"{number} {unit}"
        return number
    
    def _extract_price(self, text: str) -> Optional[str]:
        """
        Extract price constraint from text.
        Only returns hard price constraints (specific amounts).
        Qualitative terms are handled separately.
        """
        # Try numeric patterns first
        for pattern, formatter in self.price_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return formatter(match)
        return None
    
    def _extract_price_qualifier(self, text: str) -> Optional[str]:
        """
        Extract qualitative price indicators (budget, premium, luxury).
        These are not hard constraints but indicate price range preference.
        """
        text_lower = text.lower()
        for term, category in self.price_qualifiers.items():
            if re.search(rf'\b{re.escape(term)}\b', text_lower):
                return category
        return None
    
    def _extract_size(self, text: str) -> Optional[str]:
        """Extract size constraint from text."""
        for pattern, formatter in self.size_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return formatter(match)
        return None
    
    def _extract_rating(self, text: str) -> Optional[str]:
        """Extract rating constraint from text."""
        for pattern, formatter in self.rating_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return formatter(match)
        return None
    
    def _extract_from_list(self, text: str, word_list: Set[str]) -> Optional[str]:
        """Extract the first matched word from the list that appears in text."""
        text_lower = text.lower()
        for word in sorted(word_list, key=len, reverse=True):  # Longer matches first
            if re.search(rf'\b{re.escape(word)}\b', text_lower):
                return word
        return None
    
    def _extract_all_from_list(self, text: str, word_list: Set[str]) -> List[str]:
        """Extract ALL matched words from the list that appear in text."""
        text_lower = text.lower()
        matches = []
        for word in sorted(word_list, key=len, reverse=True):  # Longer matches first
            if re.search(rf'\b{re.escape(word)}\b', text_lower):
                matches.append(word)
        return matches
    #Dictionary of extracted constraints with normalized values; Single-value constraints return strings, multi-value return lists
    def extract_constraints(self, user_input: str) -> Dict[str, Union[str, List[str]]]:#this is the function that uses all prev funct
        """
        Extract specific constraints from user input.
        Supports multiple features and multi-value constraints.
        
        Args:
            user_input: The user's search query or message
            
        Returns:
            Dictionary of extracted constraints with normalized values
            Single-value constraints return strings, multi-value return lists
        """
        text = user_input.strip()
        constraints = {}
        
        # Extract price (hard constraint - specific amount)
        price = self._extract_price(text)
        if price:
            constraints['price'] = price
        
        # Extract price qualifier (soft constraint - qualitative)
        price_qualifier = self._extract_price_qualifier(text)
        if price_qualifier and 'price' not in constraints:
            # Only add if no hard price constraint exists
            constraints['price_range'] = price_qualifier
        
        # Extract size (single value)
        size = self._extract_size(text)
        if size:
            constraints['size'] = size
        
        # Extract color (can be multiple - e.g., "red and blue")
        colors = self._extract_all_from_list(text, self.colors)
        if colors:
            constraints['color'] = colors[0] if len(colors) == 1 else colors
        
        # Extract brand (single value typically)
        brand = self._extract_from_list(text, self.brands)
        if brand:
            constraints['brand'] = brand
        
        # Extract material (can be multiple)
        materials = self._extract_all_from_list(text, self.materials)
        if materials:
            constraints['material'] = materials[0] if len(materials) == 1 else materials
        
        # Extract pattern (single value typically)
        pattern = self._extract_from_list(text, self.patterns)
        if pattern:
            constraints['pattern'] = pattern
        
        # Extract fit (single value)
        fit = self._extract_from_list(text, self.fits)
        if fit:
            constraints['fit'] = fit
        
        # Extract occasion (single value)
        occasion = self._extract_from_list(text, self.occasions)
        if occasion:
            constraints['occasion'] = occasion
        
        # Extract gender (single value)
        gender = self._extract_from_list(text, self.genders)
        if gender:
            constraints['gender'] = gender
        
        # Extract category (can be multiple in some cases)
        categories = self._extract_all_from_list(text, self.categories)
        if categories:
            constraints['category'] = categories[0] if len(categories) == 1 else categories
        
        # Extract condition (single value)
        condition = self._extract_from_list(text, self.conditions)
        if condition:
            constraints['condition'] = condition
        
        # Extract features (MULTIPLE features supported)
        features = self._extract_all_from_list(text, self.features)
        if features:
            constraints['feature'] = features[0] if len(features) == 1 else features
        
        # Extract rating (single value)
        rating = self._extract_rating(text)
        if rating:
            constraints['rating'] = rating
        
        return constraints
    #Classify user intent as 'Buying' or 'Browsing' based on constraint >= 1
    def classify_intent(self, user_input: str) -> str:
        """
        Classify user intent as 'Buying' or 'Browsing' based on constraint density.
        
        Args:
            user_input: The user's search query or message
            
        Returns:
            'Buying' if ≥ 2 hard constraints found, 'Browsing' otherwise
        """
        constraints = self.extract_constraints(user_input)
        
        # Count hard constraint categories (exclude soft qualifiers)
        hard_constraints = {k: v for k, v in constraints.items() 
                           if k not in ['price_range']}
        
        # Determine intent based on hard constraint count
        if len(hard_constraints) >= 2:
            return "Buying"
        else:
            return "Browsing"
    # returns Dictionary with intent, constraint count, and extracted constraints
    def classify_with_details(self, user_input: str) -> Dict[str, Any]:
        """
        Enhanced classification that returns intent along with extracted constraints.
        
        Returns:
            Dictionary with intent, constraint count, and extracted constraints
        """
        constraints = self.extract_constraints(user_input)
        
        # Count hard constraints (exclude soft qualifiers)
        hard_constraints = {k: v for k, v in constraints.items() 
                           if k not in ['price_range']}
        
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
            "constraints": constraints,  # All constraints including soft qualifiers
            "hard_constraints": hard_constraints,  # Only hard constraints
            "query": user_input
        }
    
    def get_constraint_categories(self) -> List[str]:#"""Return list of all supported constraint categories."""
        """Return list of all supported constraint categories."""
        return [
            'price', 'price_range', 'size', 'color', 'brand', 'material', 
            'pattern', 'fit', 'occasion', 'gender', 'category', 'condition',
            'feature', 'rating'
        ]
    
    def get_feature_list(self) -> List[str]:#"""Return the complete list of supported features."""
        """Return the complete list of supported features."""
        return sorted(self.features)


if __name__ == "__main__":
    classifier = IntentClassifier()
    query = input("What would you like to search for? ")
    print("Enhanced Intent Classification with Price Qualifiers:")
    print("=" * 80)
        

    result = classifier.classify_with_details(query)
    print(f"\nQuery: '{query}'")
    print(f"Intent: {result['intent']}")
    print(f"Hard Constraint Categories: {result['constraint_count']}")
    print(f"Extracted Constraints: {result['constraints']}")
    print("-" * 80)
#test your own queries
'''
if __name__ == "__main__":
    classifier = IntentClassifier()
    query = input("What would you like to search for? ")
    print("Enhanced Intent Classification with Price Qualifiers:")
    print("=" * 80)
        

    result = classifier.classify_with_details(query)
    print(f"\nQuery: '{query}'")
    print(f"Intent: {result['intent']}")
    print(f"Hard Constraint Categories: {result['constraint_count']}")
    print(f"Extracted Constraints: {result['constraints']}")
    print("-" * 80)
'''
# Example usage and testing
'''
if __name__ == "__main__":
    classifier = IntentClassifier()
    
    # Test cases with multiple features and price variations
    test_queries = [
        "black eco-friendly leather jacket size medium under $200 waterproof",
        "Nike running shoes under $100 size 10",
        "cheap wireless headphones with bluetooth",
        "luxury leather handbag",
        "affordable summer dresses",
        "premium smartphone 5g",
        "budget gaming laptop",
        "expensive watch for men",
        "red cotton and silk dress for wedding",
        "wireless bluetooth headphones with noise cancellation under $50",
        "4K smart TV 55 inch Samsung with HDR and voice control",
        "organic cotton t-shirt eco-friendly sustainable",
        "vintage Levi's jeans size 32 waterproof stain resistant",
        "men's slim fit dress shirt size 15.5 white wrinkle-free",
        "waterproof hiking boots between $80 and $120 breathable lightweight"
    ]
    
    print("Enhanced Intent Classification with Price Qualifiers:")
    print("=" * 80)
    
    for query in test_queries:
        result = classifier.classify_with_details(query)
        print(f"\nQuery: '{query}'")
        print(f"Intent: {result['intent']}")
        print(f"Hard Constraint Categories: {result['constraint_count']}")
        print(f"Extracted Constraints: {result['constraints']}")
        print("-" * 80)
    
    # Demonstrate the specific example
    print("\n\nSpecific Example from Requirements:")
    print("=" * 80)
    query = "black eco-friendly leather jacket size medium under $200 waterproof"
    constraints = classifier.extract_constraints(query)
    print(f"Query: '{query}'")
    print(f"Extracted Constraints: {constraints}")
'''